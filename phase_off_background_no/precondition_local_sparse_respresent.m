function Result=precondition_local_sparse_respresent(img,fpos,basisList,kernelList,optparas,kernparas,debug)

%Input:
%       img: phase contrast microscopy image
%%%%%%%
%        kernparas: kernel parameter structure {'R', 'W', 'radius','dicsize'} 
%        kernparas.R: Outer radius of phase ring; 
%        kernparas.W: width of phase ring; 
%        kernparas.zetap: amplitude attenuation factors caused by phase ring. 
%        R, W and zetap are provided by microscope manufacturers
%        
%        kernpara.radius: radius of kernel
%        kernpara.dicsize': size of dictionary 
%%%%%%%
%        optparas: optimization parameter structure {'w_smooth_spatio','w_sparsity','epsilon','gamma','m_scale','maxiter','tol'}
%        optparas.w_smooth_spatio: weight of the spatial smoothness term
% 	     optparas.w_sparsity: weight of the sparsity term
% 	     optparas.epsilon: used in smooth term: (epsilon+exp)/(epsilon+1)
% 	     optparas.gamma: used in re-weighting. 1/(f+gamma): [1/(maxf+gamma), 1/gamma]
% 	     optparas.m_scale: %downsize image
% 	     optparas.maxiter: the max iteration in optimization 
%        optparas.tol: tolerance in the optimization
%        optparas.sel: maximum number of selected basis 
%%%%%%%%%
%        debug: debug model          
%Output:
%       imgproc: preconditioning result of phase contrast image 

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Example %%%%%%%%%%%%%%%%%%%%%%%%%%%
%Examples:
%Read a phase contrast image:
%  phc_img=imread('demo.tif');
% if isrgb(phc_img)
%    grayimg=rgb2gray(phc_img);
%  else
%    grayimg=phc_img;
% end
% %Convert to double precision 
%  img=im2double(grayimg);
% 
% % Optimization Parameter Setup 
%   optparas=struct('w_smooth_spatio',0.3,'w_sparsity',0.15,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',6,'tol',100);
% % Kernel Parameter Setup
%   kernparas=struct('R',4,'W',0.8,'radius',2,'dicsize',20);
%   debug=0;
%   
%  % Sparse representation of phase contrast image  
%  imgproc=precondition_sparse_respresent(cellimg,optparas,kernparas,debug);
% 
% 
% % Convert to Binary Image 
% segResult=im2bw(precond_img,graythresh(precond_img));
% %display:
% subplot(3, 1, 1), imshow(img, []), title('Original Image');
% subplot(3, 1, 2), imshow(precond_img,[]), title('Preconditioning Result');
% subplot(3, 1, 3), imshow(segResult), title('Binary Phase Constrast Image');
%%----------------------------------------------------------------------%%

%% %% Default parameters 
default_optparas=struct('w_smooth_spatio',0.3,'w_sparsity',0.15,'sel',3,'epsilon',3,'gamma',3,'m_scale',1,'maxiter',6,'tol',100);
default_kernparas=struct('R',4,'W',0.8,'radius',2,'zetap',0.8,'dicsize',20);
%% Check Input Arguement 
if(~exist('kernparas','var')),
    kernparas=default_kernparas;
end
if(~exist('optparas','var')),
    optparas=default_optparas;
end
if(~exist('debug','var'))
    debug=0;
end
if nargin<1
    error('No phase contrast image input');
end

%% Initialization 
if optparas.m_scale>1
        [nrows,ncols]=size(img);
        scale_img=img(1:optparas.m_scale:nrows,1:optparas.m_scale:ncols);
        img=scale_img;
end

[nrow,ncol]=size(img);

% initialize
% sigma = 21;
sigma = 23;
LL = zeros(size(img));
FilterSize = 241;
for jj=1:size(fpos,1);
    x = fpos(jj,1);
    y = fpos(jj,2);
    img_t = zeros(size(img));
    img_t(y,x) = 1;
    img_t = imgaussfilt(img_t,sigma,'FilterSize',FilterSize);
    LL(:,:,jj) = img_t/max(img_t(:));
end
LL = LL/max(LL(:));
[LLL II] = max(LL,[],3);
LLL = LLL.*im2bw(LLL,0.0001);
II = II.*im2bw(LLL,0.0001);
if debug
    figure(3); imshow(max(LL,[],3));
end

%% Remove constant items in phase contrast image by by a second-order polynomial surface
fprintf('Background removal\n');
orgimg = img;
% img=BackgroundRemoval(orgimg,debug);
img=BackgroundRemoval_Diffusive(orgimg,ones(size(img)),debug);

if debug
    
    figure('Name','Background')
    subplot(1,2,1)
    imshow(img)
    title('Original Image')
    subplot(1,2,2)
    imshow(img)
    title('Background Removal')
end

%% saliency Map and mask
[FX,FY] = gradient(img);
G = sqrt(FX.^2 + FY.^2);
G = G/max(G(:));

N = nrow*ncol;
[xx yy] = meshgrid(1:ncol, 1:nrow);
xx = xx(:); yy = yy(:);
X = [ones(N,1), xx, yy, xx.^2, xx.*yy, yy.^2];
    
%high pass filtering      
lowFreqRadius = 3;
lowFreq = sqrt((xx(:)-ncol/2).^2+(yy(:)-nrow/2).^2)<lowFreqRadius;
diskfilter = fspecial('disk', 10); 
myFFT = fft2(img);
myPhase = angle(myFFT);
myAmplitude = abs(myFFT);
myHighFreq = fftshift(myAmplitude);    
myHighFreq(lowFreq)=0;
saliencyMap = imfilter(abs(ifft2(ifftshift(myHighFreq).*exp(1i*myPhase))), diskfilter,'same');       

th_FCN = 0.95;
maskF = zeros(size(img));
maskF(max(LL,[],3)>th_FCN) = 1;

maskC = zeros(size(img));
maskC(saliencyMap>0.017) = 1;
maskC = imfill(maskC,'holes');
C = bwconncomp(maskC);
for ii=1:C.NumObjects;
    inds = C.PixelIdxList{ii};
    ll = maskF(inds);
    if max(ll)==0; % seed ���܂܂Ȃ�blob�͍폜
        maskC(inds) = 0;
    end
end
maskC2 = imerode(maskC,strel('disk',2));
% bw = maskC - maskC2;
% aaa = sort(img(bw>0));
% img(maskC<1) = aaa(round(length(aaa)*0.002));

%% Dictionary construction and basis selection
M=kernparas.dicsize;
K=optparas.sel;     
%selbasis=[1,6,3];

%% Gaussian Fi
im = (img-min(img(:)))/(max(img(:))-min(img(:)));
iminv = 1 - img;

[FX,FY] = gradient(im);
G = sqrt(FX.^2 + FY.^2);
G = G/max(G(:));

%% Local Sparse Representation
K = 2;
ww = (FilterSize-1)/2;
Result = zeros([size(img) K]);
for jj=1:size(fpos,1);
    pos = fpos(jj,:);
    x = fpos(jj,1);
    y = fpos(jj,2);
    tmpinds = find(II==jj);
    tmpim = zeros(size(im));
    tmpim(tmpinds) = LLL(tmpinds);
%     tmpim2 = imdilate(im2bw(tmpim,0),strel('disk',1));
%     rimg=img.*tmpim2;
%     aaaa = sort(rimg(tmpim2-im2bw(tmpim,0)>0));
%     rimg(tmpim2<=0) = aaaa(round(length(aaaa)*0.01));
%     rimg=img.*im2bw(tmpim,0);
    rimg=img; % test
    y1 = max(y-ww,1);
    y2 = min(y+ww,size(rimg,1));
    x1 = max(x-ww,1);
    x2 = min(x+ww,size(rimg,2));
    cutim = rimg(max(y-ww,1):min(y+ww,size(rimg,1)),max(x-ww,1):min(x+ww,size(rimg,2)));
    tmpcutim = tmpim(max(y-ww,1):min(y+ww,size(rimg,1)),max(x-ww,1):min(x+ww,size(rimg,2)));
    [nrow,ncol]=size(cutim);
    if size(cutim,1) ~= FilterSize || size(cutim,2) ~= FilterSize
        [tmpbasisList tmpkernelList tmpangleList] = makeBasis(cutim,kernparas,M,debug);
    else
        tmpbasisList = basisList;
        tmpkernelList = kernelList;
    end
%     rimg = cutim;
%     rimg = cutim - median(cutim(cutim>0));
    rimg = BackgroundRemoval_Diffusive2(cutim,ones(size(cutim)),debug);
    resimg = [];
    resimg2 = [];
    selbasis = [];
    candiBasis = [];
    for k=1:K;
        % Selected basis calculation 
        fprintf('Select the best basis\n');
%         selbasis(k)=BasisBestSelectionLocal(rimg,tmpim,tmpbasisList,kernelList,M,debug);
        if k==1;
            [selbasis(k) candiBasis]=BasisBestSelectionLocalFisrt(rimg,tmpcutim,tmpbasisList,tmpkernelList,M,selbasis,debug);
            candiBasis = [1:M];
        else
            selbasis(k)=BasisBestSelectionLocal(rimg,tmpcutim,tmpbasisList,tmpkernelList,M,candiBasis,ttmask,debug);
        end
        candiBasis = setdiff(candiBasis,selbasis);
        fprintf('%sth basis generation\n',num2str(k));
        kernel = tmpkernelList{selbasis(k)};
        basis = tmpbasisList{selbasis(k)};

        resfeature=basis*rimg(:); %inner product of kernel and phase contrast image 
        resfeature=reshape(resfeature,nrow,ncol);
        resfeature(resfeature<0)=0;
              
        fprintf('Calculate coefficient of the %s th basis\n',num2str(k));
%         tmpres=reshape(PhaseContrastSegParam(basis, cutim, optparas,debug),nrow,ncol); 
        tmpres=reshape(PhaseContrastSegLocalParam(basis, rimg, resfeature, tmpcutim, optparas,debug),nrow,ncol); 
        resimg(:,:,k) = tmpres;
        resimg2(:,:,k) = resfeature;
        
        fprintf('Residual error update\n');
%         rimg=rimg-reshape(basis*resimg(1+(k-1)*numel(rimg):k*numel(rimg))',[nrow,ncol]);
%         rimg=rimg-reshape(basis*tmpres(:),[nrow,ncol]);
        tttt = reshape(basis*tmpres(:),[nrow,ncol]);
        rimg2=rimg-tttt;
        inds = find((rimg>0 & rimg - tttt < 0) | (rimg<0 & rimg - tttt>0));
        tmp = imfilter(double(rimg2),fspecial('gaussian',21,3));
        inds2 = find(tmp<-0.001);
        ttmask = zeros(size(rimg));
%         ttmask(inds) = 1;
        ttmask(inds2) = 1;
        rimg = rimg2;
        
%         rimg2(inds) = 0;
%         rimg = rimg2;
%         cutim2 = cutim - 0.05;
%         contrastim = cutim2.*im2bw(cutim2,0) + 2*cutim2.*(1 - im2bw(cutim2,0));
%         contrastim(contrastim>=0) = contrastim(contrastim>=0) + 0.05;
        
        %normalization 
%         imgproc(:,:,k)=normalize(resimg(:,:,k));
%         imgproc(:,:,k)= resimg(:,:,k);
        if debug
         figure(3)
         subplot(1,2,1)
         imshow(rimg)
         title('Residual Image')
         subplot(1,2,2)
         imshow(imgproc(:,:,k))
         title(['Retoration Result for ' num2str(k) 'th basis']);
        end 
        if (norm(resimg(:,:,k))/norm(resimg(:,:,1))<0.01)
            break
        end
%         Result(y1:y2,x1:x2,k) = max(Result(y1:y2,x1:x2,k),resimg(:,:,k)).*maskC2(y1:y2,x1:x2);
        Result(y1:y2,x1:x2,k) = max(Result(y1:y2,x1:x2,k),resimg(:,:,k));
    end
%         Result(y1:y2,x1:x2,k) = max(Result(y1:y2,x1:x2,k),resfeature);
end
1;
% ndep=size(imgproc,3);
% imgproc(:,:,ndep+1:K)=0;
