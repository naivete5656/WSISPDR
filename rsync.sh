#!/bin/bash

src_dir="/home/kazuya/weakly_supervised_instance_segmentation/detection"
dest_dir="/home/kazuya/file_server/nishimura/detection"

rsync --bwlimit=5024 -avzr --delete \
    --exclude 'sequ18/' --exclude 'output/' --exclude 'text/'\
    ${src_dir} ${dest_dir}
     
     

   
