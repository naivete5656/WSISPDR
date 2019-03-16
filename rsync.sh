#!/bin/bash

src_dir="/home/kazuya/main/weakly_supervised_instance_segmentation/networks"
dest_dir="/home/kazuya/file_server2/"

rsync --bwlimit=5024 -avzr --delete \
    --exclude 'sequ18/' --exclude 'output/' --exclude 'text/'\
    ${src_dir} ${dest_dir}
     
     

   
