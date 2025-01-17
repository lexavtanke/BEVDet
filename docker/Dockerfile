# FROM nvcr.io/nvidia/tensorrt:22.07-py3
FROM nvcr.io/nvidia/tensorrt:21.06-py3

ARG CUDA=11.3
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.11.0
ARG TORCHVISION_VERSION=0.12.0
ARG ONNXRUNTIME_VERSION=1.8.1
ARG MMCV_VERSION=1.5.3
ARG PPLCV_VERSION=0.7.0
ENV FORCE_CUDA="1"

ENV DEBIAN_FRONTEND=noninteractive

# ### change the system source for installing libs
# ARG USE_SRC_INSIDE=false
# RUN if [ ${USE_SRC_INSIDE} == true ] ; \
#     then \
#         sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
#         sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
#         echo "Use aliyun source for installing libs" ; \
#     else \
#         echo "Keep the download source unchanged" ; \
#     fi

# ### update apt and install libs
# RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN chmod 777 /tmp
RUN apt-get clean && apt-get update &&\
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libopencv-dev libspdlog-dev --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

RUN pip3 install pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja  libtiff pillow 

### pytorch
# RUN pip3 install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit==${CUDA} 
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
 
### install mmcv-full
RUN pip3 install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html  
WORKDIR /root/workspace
# ### get onnxruntime
# RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
#     && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
#     pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple

# ### cp trt from pip to conda
# RUN cp -r /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* /opt/conda/lib/python${PYTHON_VERSION}/site-packages/

# ### install mmdeploy
# ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
# ENV TENSORRT_DIR=/workspace/tensorrt
# ARG VERSION
# RUN git clone https://github.com/HuangJunJie2017/mmdeploy.git
# RUN cd mmdeploy &&\
#     if [ -z ${VERSION} ] ; then echo "No MMDeploy version passed in, building on master" ; else git checkout tags/v${VERSION} -b tag_v${VERSION} ; fi &&\
#     git submodule update --init --recursive &&\
#     mkdir -p build &&\
#     cd build &&\
#     cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" .. &&\
#     make -j$(nproc) &&\
#     cd .. &&\
#     pip install -e .  -i https://pypi.tuna.tsinghua.edu.cn/simple

# ### build sdk
# RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
#     cd ppl.cv &&\
#     git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
#     ./build.sh cuda

# ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

# RUN cd /root/workspace/mmdeploy &&\
#     rm -rf build/CM* build/cmake-install.cmake build/Makefile build/csrc &&\
#     mkdir -p build && cd build &&\
#     cmake .. \
#         -DMMDEPLOY_BUILD_SDK=ON \
#         -DMMDEPLOY_BUILD_EXAMPLES=ON \
#         -DCMAKE_CXX_COMPILER=g++ \
#         -Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
#         -DTENSORRT_DIR=${TENSORRT_DIR} \
#         -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
#         -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
#         -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
#         -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
#         -DMMDEPLOY_CODEBASES=all &&\
#     make -j$(nproc) && make install &&\
#     export SPDLOG_LEVEL=warn &&\
#     if [ -z ${VERSION} ] ; then echo "Built MMDeploy master for GPU devices successfully!" ; else echo "Built MMDeploy version v${VERSION} for GPU devices successfully!" ; fi

# ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

RUN pip3 install mmdet==2.25.1 mmsegmentation==0.25.0  && \
    cd ..

RUN pip3 install pycuda \
    lyft_dataset_sdk \
    networkx==2.2 \
    numba==0.56.4 \
    numpy \
    nuscenes-devkit \
    plyfile \
    scikit-image \
    tensorboard \
    transforms3d \
    trimesh==2.35.39 

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros2/ubuntu focal main" > /etc/apt/sources.list.d/ros2-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO galactic

# install ros2 packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    ros-galactic-ros-base ros-galactic-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]