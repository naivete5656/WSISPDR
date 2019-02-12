%%%%%%% Demo of sparse representation for phase contrast image
%     kernparas: kernel parameter structure {'R', 'W', 'radius','zetap','dicsize'} 
%        -- R: Outer radius of phase ring; 
%        -- W: width of phase ring; 
%        -- zetap: amplitude attenuation factors caused by phase ring.
%        -- R, W and zetap are provided by microscope manufacturers
%        -- radius: radius of kernel
%        -- dicsize: size of dictionary 
%%--------------------------------------------------------------------------------------------%
%      optparas: optimization parameter structure {'w_smooth_spatio','w_sparsity','epsilon','gamma','m_scale','maxiter','tol'}
%        --w_smooth_spatio: weight of the spatial smoothness term
% 	     --w_sparsity: weight of the sparsity term
% 	     --epsilon: used in smooth term: (epsilon+exp)/(epsilon+1)
% 	     --gamma: used in re-weighting. 1/(f+gamma): [1/(maxf+gamma), 1/gamma]
% 	     --m_scale: %downsize image
% 	     --maxiter: the max iteration in optimization 
%        --tol: tolerance in the optimization
%        --sel: maximum number of selected basis 
%%--------------------------------------------------------------------------------------------%
%     mode: algorithm select  
%        --'linear_model': an earlier algorithm published on Medical Image Analysis (2012), 
%                          which restore dark cells in phase contrast images
%        --'sparse _respresent': a recent algorithm published on MICCAI2012, which restore phase contrast images
%                                with sprarse representation model
%%--------------------------------------------------------------------------------------------%
%        debug: debug model     
%
%%------------------------------------------Output--------------------------------------------------%
%
%       precd_img: preconditioning result of phase contrast image 

%%%%Reference:
% [1] Hang Su, Zhaozheng Yin, Takeo Kanade, Seungil Huh: Phase Contrast Image Restoration via Dictionary Representation of Diffraction Patterns.
% the 15th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2012: 615-622

% [2] Zhaozheng Yin, Takeo Kanade, Mei Chen: Understanding the phase
% contrast optics to restore artifact-free microscopy images for segmentation. Medical Image Analysis 16(5): 1047-1062 (2012)

% [3] http://www.celltracking.ri.cmu.edu/

% Composed by Zhaozheng Yin, Hang Su and modified by Seung-il Huh 
% Robotics Institute, Carnegie Mellon University  
% ------------------------------------------------------------------------------------------------------------
% If you have any suggestions, questions, and bug reports etc, please feel free
% to contact Hang Su at (suhangss@gmail.com)

% Copyright (C) Zhaozheng Yin, Hang Su
% All rights reserved.

%%-------------------------------------Recommended Parameter--------------------------------------------------------------%%
% For the human-stem-cells (HSC), provided by Pitt University
% kernparas=struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);
% optparas=struct('w_smooth_spatio',1,'w_sparsity',0.4,'sel',20,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',eps);
%  
% For Bovine aortic endothelial cell (BAEC)
% kernparas = struct('R',4000,'W',800,'radius',5,'zetap',0.8,'dicsize',20);
% optparas=struct('w_smooth_spatio',1,'w_sparsity',0.4,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',10);
%% -------------------------------------------------------------------------------------------------------------------------%%

clear all
close all
clc

%% Read a phase contrast image:
addpath('./func')

ori_path = sprintf('../images/sequence/sequ17/ori/*.tif');
ori_paths = dir(ori_path);

network_pre_path = sprintf('../images/sequence/sequ17/9/*.tif');
network_pre_paths = dir(network_pre_path);

for frame = 1:length(ori_paths)
    ori_path = ori_paths(frame);
    network_pre_path = network_pre_paths(frame);
    phc_img = imread([ori_path.folder,'/',ori_path.name]);
    if size(phc_img,3) > 1
       grayimg=rgb2gray(phc_img);
    else
       grayimg=phc_img;
    end
    %Convert to double precision 
%     img=double(grayimg)/2^12;
    img=double(grayimg)/4096;

    %% Read a FCN result:
    FCN = imread([network_pre_path.folder,'/',network_pre_path.name]);
    FCN = FCN(:,:,1);
    FCN = double(FCN)/255;
    FCN = imresize(FCN,size(img));

    % foreground seed
    th_FCN = 0.8;
    maskF = zeros(size(img));
    maskF(FCN>th_FCN) = 1;

    fpos = [];
    C= bwconncomp(maskF);
    stats = regionprops(C,'centroid');
    for ii=1:C.NumObjects
        fpos(ii,:) = round(stats(ii).Centroid);
    end

    %% Parameter Setup 
    
    optparas=struct('w_smooth_spatio',1,'w_sparsity',0.005,'w_prior',0.2,'sel',3,'epsilon',1,'gamma',0.1,'m_scale',0.5,'maxiter',100,'tol',eps);
    kernparas=struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);
    
    % Algorithm mode
    % mode='sparse_respresent';
    mode='local_model';
    debug=0;

    %% Precondition of phase contrast image
    % make basis and kernel
    FilterSize = 241;
    M = kernparas.dicsize;
    tmpimg = zeros(FilterSize,FilterSize);
    [basisList kernelList angleList] = makeBasis(tmpimg,kernparas,M,debug);
    
    precd_img=precondition_local_phase_contrast(img,fpos,basisList,kernelList,optparas,kernparas,mode,debug);
    
    % normalize
    precd_img=normalize(precd_img);
    
    precd_img = (precd_img-min(precd_img(:)))/(max(precd_img(:))-min(precd_img(:)));
    
    precd_img = max(precd_img,[],3);
    
%     figure(1),imshow(precd_img);
    
    save_path = sprintf('../outputs/phase-off/sequ17/%05d.tif',frame-1);
    imwrite(precd_img,save_path);
end
















