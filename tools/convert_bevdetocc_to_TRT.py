import sys
sys.path.insert(0, '/root/workspace/BEVDET')

import argparse

import torch
import torch.onnx
from onnxruntime.tools import pytorch_export_contrib_ops
from mmcv import Config
from mmdeploy.backend.tensorrt.utils import save, search_cuda_version

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import os
from typing import Dict, Optional, Sequence, Union

import h5py
import mmcv
import numpy as np
import onnx
import pycuda.driver as cuda
import tensorrt as trt
import torch
import tqdm
from mmcv.runner import load_checkpoint
from mmdeploy.apis.core import no_mp
from mmdeploy.backend.tensorrt.calib_utils import HDF5Calibrator
from mmdeploy.backend.tensorrt.init_plugins import load_tensorrt_plugin
from mmdeploy.utils import load_config
from packaging import version
from torch.utils.data import DataLoader

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module


class HDF5CalibratorBEVDet(HDF5Calibrator):

    def get_batch(self, names: Sequence[str], **kwargs) -> list:
        """Get batch data."""
        if self.count < self.dataset_length:
            if self.count % 100 == 0:
                print('%d/%d' % (self.count, self.dataset_length))
            ret = []
            for name in names:
                input_group = self.calib_data[name]
                if name == 'img':
                    data_np = input_group[str(self.count)][...].astype(
                        np.float32)
                else:
                    data_np = input_group[str(self.count)][...].astype(
                        np.int32)

                # tile the tensor so we can keep the same distribute
                opt_shape = self.input_shapes[name]['opt_shape']
                data_shape = data_np.shape

                reps = [
                    int(np.ceil(opt_s / data_s))
                    for opt_s, data_s in zip(opt_shape, data_shape)
                ]

                data_np = np.tile(data_np, reps)

                slice_list = tuple(slice(0, end) for end in opt_shape)
                data_np = data_np[slice_list]

                data_np_cuda_ptr = cuda.mem_alloc(data_np.nbytes)
                cuda.memcpy_htod(data_np_cuda_ptr,
                                 np.ascontiguousarray(data_np))
                self.buffers[name] = data_np_cuda_ptr

                ret.append(self.buffers[name])
            self.count += 1
            return ret
        else:
            return None


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='work dir to save file')
    parser.add_argument(
        '--prefix', default='bevdet4d_occ', help='prefix of the save file name')
    parser.add_argument(
        '--fp16', action='store_true', help='Whether to use tensorrt fp16')
    parser.add_argument(
        '--int8', action='store_true', help='Whether to use tensorrt int8')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def create_calib_input_data_impl(calib_file: str,
                                 dataloader: DataLoader,
                                 model_partition: bool = False,
                                 metas: list = []) -> None:
    with h5py.File(calib_file, mode='w') as file:
        calib_data_group = file.create_group('calib_data')
        assert not model_partition
        # create end2end group
        input_data_group = calib_data_group.create_group('end2end')
        input_group_img = input_data_group.create_group('img')
        input_keys = [
            'ranks_bev', 'ranks_depth', 'ranks_feat', 'interval_starts',
            'interval_lengths'
        ]
        input_groups = []
        for input_key in input_keys:
            input_groups.append(input_data_group.create_group(input_key))
        metas = [
            metas[i].int().detach().cpu().numpy() for i in range(len(metas))
        ]
        for data_id, input_data in enumerate(tqdm.tqdm(dataloader)):
            # save end2end data
            input_tensor = input_data['img_inputs'][0][0]
            input_ndarray = input_tensor.squeeze(0).detach().cpu().numpy()
            # print(input_ndarray.shape, input_ndarray.dtype)
            input_group_img.create_dataset(
                str(data_id),
                shape=input_ndarray.shape,
                compression='gzip',
                compression_opts=4,
                data=input_ndarray)
            for kid, input_key in enumerate(input_keys):
                input_groups[kid].create_dataset(
                    str(data_id),
                    shape=metas[kid].shape,
                    compression='gzip',
                    compression_opts=4,
                    data=metas[kid])
            file.flush()


def create_calib_input_data(calib_file: str,
                            deploy_cfg: Union[str, mmcv.Config],
                            model_cfg: Union[str, mmcv.Config],
                            model_checkpoint: Optional[str] = None,
                            dataset_cfg: Optional[Union[str,
                                                        mmcv.Config]] = None,
                            dataset_type: str = 'val',
                            device: str = 'cpu',
                            metas: list = [None]) -> None:
    """Create dataset for post-training quantization.

    Args:
        calib_file (str): The output calibration data file.
        deploy_cfg (str | mmcv.Config): Deployment config file or
            Config object.
        model_cfg (str | mmcv.Config): Model config file or Config object.
        model_checkpoint (str): A checkpoint path of PyTorch model,
            defaults to `None`.
        dataset_cfg (Optional[Union[str, mmcv.Config]], optional): Model
            config to provide calibration dataset. If none, use `model_cfg`
            as the dataset config. Defaults to None.
        dataset_type (str, optional): The dataset type. Defaults to 'val'.
        device (str, optional): Device to create dataset. Defaults to 'cpu'.
    """
    with no_mp():
        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load cfg if necessary
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load dataset_cfg if necessary
        dataset_cfg = load_config(dataset_cfg)[0]

        from mmdeploy.apis.utils import build_task_processor
        task_processor = build_task_processor(model_cfg, deploy_cfg, device)

        dataset = task_processor.build_dataset(dataset_cfg, dataset_type)

        dataloader = task_processor.build_dataloader(
            dataset, 1, 1, dist=False, shuffle=False)

        create_calib_input_data_impl(
            calib_file, dataloader, model_partition=False, metas=metas)


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              output_file_prefix: str,
              input_shapes: Dict[str, Sequence[int]],
              max_workspace_size: int = 0,
              fp16_mode: bool = False,
              int8_mode: bool = False,
              int8_param: Optional[dict] = None,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              **kwargs) -> trt.ICudaEngine:
    """Create a tensorrt engine from ONNX.

    Modified from mmdeploy.backend.tensorrt.utils.from_onnx
    """

    import os
    old_cuda_device = os.environ.get('CUDA_DEVICE', None)
    os.environ['CUDA_DEVICE'] = str(device_id)
    import pycuda.autoinit  # noqa:F401
    if old_cuda_device is not None:
        os.environ['CUDA_DEVICE'] = old_cuda_device
    else:
        os.environ.pop('CUDA_DEVICE')

    load_tensorrt_plugin()
    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    cuda_version = search_cuda_version()
    if cuda_version is not None:
        version_major = int(cuda_version.split('.')[0])
        if version_major < 11:
            # cu11 support cublasLt, so cudnn heuristic tactic should disable CUBLAS_LT # noqa E501
            tactic_source = config.get_tactic_sources() - (
                1 << int(trt.TacticSource.CUBLAS_LT))
            config.set_tactic_sources(tactic_source)

    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        if version.parse(trt.__version__) < version.parse('8'):
            builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        assert int8_param is not None
        config.int8_calibrator = HDF5CalibratorBEVDet(
            int8_param['calib_file'],
            input_shapes,
            model_type=int8_param['model_type'],
            device_id=device_id,
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2))
        if version.parse(trt.__version__) < version.parse('8'):
            builder.int8_mode = int8_mode
            builder.int8_calibrator = config.int8_calibrator

    # create engine
    engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'

    save(engine, output_file_prefix + '.engine')
    return engine


def main():
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    load_tensorrt_plugin()
    assert 'bev_pool_v2' in get_plugin_names(), \
        'bev_pool_v2 is not in the plugin list of tensorrt, ' \
        'please install mmdeploy from ' \
        'https://github.com/HuangJunJie2017/mmdeploy.git'

    if args.int8:
        assert args.fp16
    model_prefix = args.prefix
    if args.int8:
        model_prefix = model_prefix + '_int8'
    elif args.fp16:
        model_prefix = model_prefix + '_fp16'
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'

    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    # build the dataloader
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation=True
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # print("grid_size")
    # print(model.img_view_transformer.grid_size[0])
    # print(model.img_view_transformer.grid_size[1])
    # print(model.img_view_transformer.grid_size[2])
    assert model.img_view_transformer.grid_size[0] == 200 #128
    assert model.img_view_transformer.grid_size[1] == 200 #128
    assert model.img_view_transformer.grid_size[2] == 16 #1
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse'
        model = fuse_module(model)
    model.cuda()
    model.eval()

    for i, data in enumerate(data_loader):
        # print('__________________')
        # print(f'data type is {type(data)} data len is {len(data)}')
        # print(f'keys in data dict are {list(data.keys())}')
        # print(f'img_metas type is {data["img_metas"]}')
        # print(f'img_inputs type is {type(data["img_inputs"])} img_inputs len is {len(data["img_inputs"])}')

        
        inputs = [t.cuda() for t in data['img_inputs'][0]]
        # print('==================')
        # print(len(inputs))
        # prepare inputs for firing weight of the model 
        metas = model.get_bev_pool_input(inputs)
        
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = model.prepare_inputs(inputs, stereo=True)
        
        img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[0], sensor2keyegos[0], ego2globals[0], intrins[0], \
                post_rots[0], post_trans[0]
        
        mlp_input = model.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
        # print(f'mlp_input shape is {mlp_input.shape}')
        feat_prev_iv = None
        extra_ref_frame = False 
        inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[0],
                               extra_ref_frame)
        
        
        with torch.no_grad():
            feat_prev, depth, stereo_feat_prev = model.prepare_bev_feat(*inputs_curr)
        # print(f'feat_prev shape is {feat_prev.shape}')
        # print(f'stereo_feat_prev shape is {stereo_feat_prev.shape}')
        # print(f'curr2adjsensor[0] shape is {curr2adjsensor[0].shape}')
        img = inputs[0].squeeze(0)
        # img = inputs[0]
        # print(f'img shape before callting model is {img.shape}')
        # print(f'before squeeze {inputs[0].shape}')

        stereo_metas = {'cv_feat_list' : [stereo_feat_prev, stereo_feat],
                'post_trans' : post_tran,
                'post_rots' : post_rot,  
                'k2s_sensor' : k2s_sensor,
                'intrins' : intrin,
                'frustum' : model.img_view_transformer.cv_frustum.to(x)}


        cost_volumn = model.img_view_transformer.depth_net.calculate_cost_volum()
        
        with torch.no_grad():
            # pytorch_export_contrib_ops.register()
            torch.onnx.export(
                model,
                (img.float().contiguous(), feat_prev.float().contiguous(),
                 stereo_feat_prev.float().contiguous(), mlp_input.float().contiguous(),
                 post_tran.float().contiguous(), post_rot.float().contiguous(),
                 curr2adjsensor[0].float(), intrin.float().contiguous(),
                 metas[1].int().contiguous(),
                 metas[2].int().contiguous(), metas[0].int().contiguous(),
                 metas[3].int().contiguous(), metas[4].int().contiguous()),
                args.work_dir + model_prefix + '.onnx',
                opset_version=11,
                input_names=[
                    'img', 'feat_prev', 'stereo_feat_prev', 'mlp_input',
                    'post_tran', 'post_rot', 'curr2adjsensor', 'intrin',
                    'ranks_depth', 'ranks_feat', 'ranks_bev',
                    'interval_starts', 'interval_lengths'
                ],
                output_names=[f'output_{j}' for j in
                              range(3)])
        break
    # check onnx model
    onnx_model = onnx.load(args.work_dir + model_prefix + '.onnx')
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print('ONNX Model Incorrect')
    else:
        print('ONNX Model Correct')

    # convert to tensorrt
    num_points = metas[0].shape[0]
    num_intervals = metas[3].shape[0]
    img_shape = img.shape
    feat_prev_shape = feat_prev.shape
    # print('input_shapes')
    input_shapes = dict(
        img=dict(
            min_shape=img_shape, opt_shape=img_shape, max_shape=img_shape),
        feat_prev=dict(
            min_shape=feat_prev_shape, opt_shape=feat_prev_shape, max_shape=feat_prev_shape),
        ranks_depth=dict(
            min_shape=[num_points],
            opt_shape=[num_points],
            max_shape=[num_points]),
        ranks_feat=dict(
            min_shape=[num_points],
            opt_shape=[num_points],
            max_shape=[num_points]),
        ranks_bev=dict(
            min_shape=[num_points],
            opt_shape=[num_points],
            max_shape=[num_points]),
        interval_starts=dict(
            min_shape=[num_intervals],
            opt_shape=[num_intervals],
            max_shape=[num_intervals]),
        interval_lengths=dict(
            min_shape=[num_intervals],
            opt_shape=[num_intervals],
            max_shape=[num_intervals]))
    print('deploy_cfg')
    deploy_cfg = dict(
        backend_config=dict(
            type='tensorrt',
            common_config=dict(
                fp16_mode=args.fp16,
                max_workspace_size=1073741824,
                int8_mode=args.int8),
            model_inputs=[dict(input_shapes=input_shapes)]),
        codebase_config=dict(
            type='mmdet3d', task='VoxelDetection', model_type='end2end'))

    if args.int8:
        print('if int8')
        calib_filename = 'calib_data.h5'
        calib_path = os.path.join(args.work_dir, calib_filename)
        create_calib_input_data(
            calib_path,
            deploy_cfg,
            args.config,
            args.checkpoint,
            dataset_cfg=None,
            dataset_type='val',
            device='cuda:0',
            metas=metas)
    
    print('from_onnx')
    from_onnx(
        args.work_dir + model_prefix + '.onnx',
        args.work_dir + model_prefix,
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        int8_param=dict(
            calib_file=os.path.join(args.work_dir, 'calib_data.h5'),
            model_type='end2end'),
        max_workspace_size=1 << 30,
        input_shapes=input_shapes)

    if args.int8:
        os.remove(calib_path)


if __name__ == '__main__':

    main()