#!/bin/bash

src_dir="/home/kazuya/weakly_supervised_instance_segmentation/graphcut/"
dest_dir="/home/kazuya/file_server/nishimura/graphcut"

rsync --bwlimit=5024 -avzr --delete \
    --exclude 'weight/' --exclude 'output/' --exclude 'text/'\
    ${src_dir} ${dest_dir}
     
     

   
