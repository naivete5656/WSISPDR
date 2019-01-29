#!/bin/bash

src_dir="/home/kazuya/weakly_supervised_instance_segmentation/unet_pytorch/"
dest_dir="/home/kazuya/file_server/nishimura/unet_pytorch"

rsync --bwlimit=5024 -avzr --delete \
    --exclude 'weight/' --exclude 'output/' --exclude 'text/'\
    ${src_dir} ${dest_dir}
     
     

   
