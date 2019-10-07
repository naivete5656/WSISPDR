#!/bin/bash

docker run --runtime=nvidia --rm -it --name root -v $(pwd):/workdir -w /workdir pytorch /bin/bash