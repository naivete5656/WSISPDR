#!/bin/bash

docker run --runtime=nvidia --rm -it -p 8888:8888 --name root -v $(pwd):/workdir -e PASSWORD=humanif -w /workdir pytorch