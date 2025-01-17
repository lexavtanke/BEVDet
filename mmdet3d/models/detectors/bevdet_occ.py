# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D

import torch
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv3
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np


@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

    def loss_single(self,voxel_semantics,mask_camera,preds):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        print('BEVStereo4DOCC simple_test')
        print(f'input img len is {len(img)}')
        print(f'input img shape is {img[0].shape}')
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        print(f'img_feats shape is {img_feats[0].shape}')
        print('occ_pred will happens next')
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)# bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        print(f'before final_conv img_feats[0].shape {img_feats[0].shape}')
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses


@DETECTORS.register_module()
class BEVStereo4DOCCTRT(BEVStereo4DOCC):
    def forward(
        self,
        img,
        feat_prev,
        cost_volumn,
        mlp_input,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        # actually logic is to get batch of images 
        # extract features from each of them 
        # alight features with ego transform information
        # concatenate 


        # print("_____________forward method_____________")
        # print(f'input img shape {img.shape}')

        # x, stereo_feat = self.image_encoder(img, stereo=True)
        # example from bevdet
        x = self.img_backbone(img[:6]) # first 6 images - first frame of batch
        stereo_feat = x[0]
        x = x[1:]
        x = self.img_neck(x) # compress img feature
        # print(f'after image_encoder shape {x.shape}')
        # print(f'stereo_feat shape {stereo_feat.shape}')
        # print(f'stereo_feat_prev shape {stereo_feat_prev.shape}')
        # print(f'k2s_sensor shape {k2s_sensor.shape}')
        # stereo_metas = {'cv_feat_list' : [stereo_feat_prev, stereo_feat],
        #                 'post_trans' : post_tran,
        #                 'post_rots' : post_rot,  
        #                 'k2s_sensor' : k2s_sensor,
        #                 'intrins' : intrin,
        #                 'frustum' : self.img_view_transformer.cv_frustum.to(x)}
        
        # x = self.img_view_transformer.depth_net(x, mlp_input, stereo_metas) 
       
       # estimate depth
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)

        cost_volumn = self.img_view_transformer.depth_net.cost_volumn_net(cost_volumn)
        depth = torch.cat([depth, cost_volumn], dim=1)     
        depth = self.depth_conv(depth)



        x = torch.cat([depth, context], dim=1)

        # print(f'after depth_net s shape {x.shape}')
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        # print(f'after depth_net depth shape {depth.shape}')

        # transform feature
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        # print(f'after tran_feat shape {tran_feat.shape}')
        
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        # print(f'after permute shape {tran_feat.shape}')
        # apply pooling operation
        x = TRTBEVPoolv3.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths, 
                               200, 200)
        # x = x.permute(0, 3, 1, 2).contiguous()
        # print(f'after x.permute shape {x.shape}')

        bev_feat = self.pre_process_net(x)[0]
        # bev_feat = x
        curr_bev_feat = bev_feat # should be in output for using in next frame 
        # print(f'after curr_bev_feat shape {curr_bev_feat.shape}')
        
        bev_feat_list = []
        bev_feat_list.append(curr_bev_feat)
        # _, C, H, W = feat_prev.shape
        # bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))
        bev_feat_list.append(feat_prev)        
        #cat prev features with current features
        bev_feat = torch.cat(bev_feat_list, dim=1)
        # apply bev_encoder
        bev_feat = self.bev_encoder(bev_feat)
        bev_feat = bev_feat.unsqueeze(0)
        # print(f'after bev_encoder shape {bev_feat.shape}')
        

        # occ head 
        # print(f'before final_conv bev_feat[0.shape] {bev_feat[0].shape}')
        occ_pred = self.final_conv(bev_feat[0]).permute(0, 4, 3, 2, 1)
        occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0) # convertion to np.uint8 should be done 
        print('ready for return')
        print(f'occ_res type is {type(occ_res)}')
        return occ_res, curr_bev_feat, stereo_feat
    
    
    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        # print('________________________________')
        # print(f'type is {type(input[0][0])}')
        # print(input[0][0].shape)
        # for inp in input:
        #     print(f'input type is {type(inp)}')
        # print(f'input len is {len(input)}')
        # coor = self.img_view_transformer.get_lidar_coor(*input[1:7]) # error because list 
        coor = self.img_view_transformer.get_lidar_coor(*[x[0] for x in input[1:7]])
        # return meta data for formward method ranks 
        # input[1] sensor2keyegos
        # input[6] bda
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor) #, input[1], input[6]


        