#!/bin/bash

cd /home/file_server/nishimura/unet_pytorch

docker run --runtime=nvidia \
     --rm -it \
     -p 8888:8888 \
     --name root \
     -v $(pwd):/workdir \
     -e PASSWORD=humanif \
     -w /workdir pytorch