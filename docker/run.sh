#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ] ; do SOURCE="$(readlink "$SOURCE")"; done
DOCKER_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
REPO_ROOT="$( cd -P "$( dirname "$DOCKER_DIR" )" && pwd )"

DOCKER_VERSION=$(docker version -f "{{.Server.Version}}")
DOCKER_MAJOR=$(echo "$DOCKER_VERSION"| cut -d'.' -f 1)

if [ "${DOCKER_MAJOR}" -ge 19 ]; then
    runtime="--gpus=all"
else
    runtime="--runtime=nvidia"
fi

# no need to do `xhost +` anymore
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# specify the image, very important
IMAGE="bevdet:latest"
RUNTIME=$runtime

docker run ${RUNTIME} -it --rm \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
            --volume=${REPO_ROOT}/:/root/workspace/BEVDET \
            --volume=/home/alex/datasets/nuscenes:/root/workspace/BEVDET/data/nuscenes \
            --shm-size 16g \
	    --entrypoint /bin/bash \
            ${IMAGE}
