<h2 align="center">Weakly Supervised Cell Instance Segmentation<br>by Propagating from Detection Response</h2>

[![](https://img.shields.io/badge/pytorch-1.0-red.svg)](https://github.com/ZhouYanzhao/PRM/tree/pytorch)

[[Home]](http://human.ait.kyushu-u.ac.jp/~bise/researches-bise.html) [[Project]](https://naivete5656.github.io/WSISPDR/) 
<!-- [[Paper]](https://arxiv.org/pdf/1804.00880)  -->
<!-- [[Supp]](http://yzhou.work/PRM/Supplementary.pdf)  -->
<!-- [[Poster]](http://yzhou.work/PRM/Poster.pdf)  -->
<!-- [[Presentation]](https://www.youtube.com/embed/lNqXyJliVSo?start=4615&end=4850&autoplay=1&controls=0) -->

![Illustration](./image/proposed_method_overview.png)

## Prerequisites
- python3
- ubuntu 18.04
- CPU or GPU
- nvidia driver 430
- matlab

## Installation

The setting of Detection network with python 
### Conda user
```bash
conda env create -f=env_name.yml
conda activate pytorch
```

### Docker user
```besh
docker build ./docker
sh run_docker.sh
```
### Graph-cut installation
Graph-cut setting

We use following code.

https://jp.mathworks.com/matlabcentral/fileexchange/38555-kernel-graph-cut-image-segmentation
Copyright (c) 2012, Ismail Ben Ayed
All rights reserved.

```bash
mkdir graphcut 
cd graphcut
wget http://www.wisdom.weizmann.ac.il/~bagon/matlab_code/GCmex1.9.tar.gz
tar -zxvf GCmex1.9.tar.gz
matlab -nodesktop -nosplash -r 'compile_gc; exit'
cd ..
```


## Demo
## Back propagate from each cell
### Use cuda
```bash
python propagate_main.py -g
```
### Use cpu
```bash
python detection_train.py 
```
#### Optins:
-i :input path(str)

-o :output path(str)

-w :weight path want to load

-g :whether use CUDA

## Graph-cut
```bash
matlab -nodesktop -nosplash -r 'graphcut; exit'
```

<div style="color:#0000FF" align="center">
 <img src="./image/test/ori/00000.png" width="290"/> <img src="./image/test/gt/00000.png" width="290"/><img src="./output/seg/result_bp/00000segbp.png" width="290"/>
</div>
**This is a sample code.**
**We don't provide dataset.**
If you want to apply your dataset, you should prepare the original image and point level annotation(cell centroid).
The text file format contains a cell position(frame,x,y) as each row.
Prepare the same format text file for your dataset.

## Generate likelyfood map
```bash
python likelymapgen.py 
```
#### Option:
-i :txt_file_path (str)

-o :output_path  (str)

-w :width (int)

-h :height (int)

-g :gaussian variance size (int)


## Train 
### Use cuda
'''bash
python detection_train.py -g
'''
### Use cpu
'''bash
python detection_train.py 
'''
#### Optins:
-t :train path(str)

-v :validation path(str)

-w :save path of weight(str)

-g :whether use CUDA

-b :batch size (default is 16)

-e :epochs (default is 500)

-l :learning rate(default is 1e-3)

## Predict
### Use cuda
'''bash
python detection_train.py -g
'''
### Use cpu
'''bash
python detection_train.py 
'''
#### Optins:
-i :input path(str) 

-o :output path(str)

-w :weight path want to load

-g :whether use CUDA

