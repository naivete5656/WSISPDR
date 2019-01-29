function resimg=precondition_local_phase_contrast(img,fpos,basisList,kernelList,optparas,kernparas,mode,debug)
%Input:
%     img: phase contrast microscopy image
%%--------------------------------------------------------------------------------------------%
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
%        --'linear_model': an earlier algorithm published on MICCAI2010, which restore dark cells in phase contrast images
%        --'sparse _respresent': a recent algorithm published on MICCAI2012, which restore phase contrast images with sprarse representation model
%%--------------------------------------------------------------------------------------------%
%        debug: debug model     
%
%%------------------------------------------Output--------------------------------------------------%
%
%       precd_img: preconditioning result of phase contrast image 

%%%%Reference:
% [1] Hang Su, Zhaozheng Yin, Takeo Kanade, Seungil Huh: Phase Contrast Image Restoration via Dictionary Representation of Diffraction Patterns. the 15th 
%     International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2012: 615-622
%
% [2] Zhaozheng Yin, Takeo Kanade, Mei Chen: Understanding the phase
% contrast optics to restore artifact-free microscopy images for segmentation. Medical Image Analysis 16(5): 1047-1062 (2012)

% [3] http://www.celltracking.ri.cmu.edu/

%Composed by Hang Su on 09/25/2012 
% Robotics Institute, Carnegie Mellon University  
%If you have any suggestions, questions, and bug reports etc please feel free
%to contact Hang Su at (suhangss@gmail.com)

% Copyright (C) Hang Su
% All rights reserved.

%% %% Default parameters 
default_optparas=struct('w_smooth_spatio',0.3,'w_sparsity',0.15,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',100,'tol',10);
default_kernparas=struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);
%% Check Input Arguement 
if(~exist('kernparas','var')),
    kernparas=default_kernparas;
end
if(~exist('optparas','var')),
    optparas=default_optparas;
end
if(~exist('debug','var')),
    debug=0;
end
if(~exist('mode','var')),
    mode='sparse_represent_seg';
end
if nargin<1
    error('No phase contrast image input');
end

%% Check validity of input paramters
if ~checkOptParaValid(optparas)
    error('Illegal Optimization Parameter Input.');
end
if ~checkKernParaValid(kernparas)
    error('Illegal Kernel Parameter Input');
end
%% 
fprintf('begin preconditioning\n');
if  strcmp(mode,'sparse_respresent')
    resimg=precondition_sparse_respresent(img,optparas,kernparas,debug);
elseif strcmp(mode,'local_model')
    resimg=precondition_local_sparse_respresent(img,fpos,basisList,kernelList,optparas,kernparas,debug);
elseif strcmp(mode,'linear_model')
    resimg=precondition_linear_model(img,optparas,kernparas,debug);
end

























