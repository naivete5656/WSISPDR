addpath('C:\Users\bise-r\Desktop\code\toolbox\Kernel_GraphCuts\GCMex-master\')
addpath('C:\Users\bise-r\Desktop\code\toolbox\DRLSE_v0\')
debug = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load data
baseID = '00000';
baseID = '00005';
baseID = '00015';
% baseID = '00017';
baseID = '2983';
% baseID = '2984';
basefolder = ['..\graphcut_data\' baseID '\'];
% basefolder = ['..\backprop\' baseID '\'];

infile = fullfile(basefolder,'original_image.tif');
prefile = fullfile(basefolder,'preconditioning2.tif');
fcnfile = fullfile(basefolder,'detection.tif');
posfile = fullfile(basefolder,[baseID '.txt']);
bpfolder = [basefolder 'each_peak_backward\'];
bpfiles = dir([bpfolder '*.tif']);
% posfile = fullfile(basefolder,'peaks.txt');
% bpfolder = [basefolder 'each_peak\'];
% bpfiles = dir([bpfolder '*.mat']);

orgim = imread(infile);
preim = imread(prefile);
F = imread(fcnfile);
orgim = double(orgim)/255;
preim = double(preim)/255;
F = double(F)/255;
fpos = readtable(posfile);
fpos = fpos.Variables;
fpos = fpos(:,[3 2]); % [y x]

[Ny Nx] = size(orgim);
Nz = length(bpfiles);

BP = zeros(Ny,Nx,Nz);
for fidx=1:length(bpfiles);
    bpfile = fullfile(bpfolder,bpfiles(fidx).name);
    bp = imread(bpfile);
    BP(:,:,fidx) = double(bp)/255;
%     load(bpfile);
%     BP(:,:,fidx) = double(image)/255;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% preprocess
% detect seed
th_FCN = 0.91;
maskF = zeros(size(F));
maskF(F>th_FCN) = 1;

figure(1);imshow(F)
hold on; scatter(fpos(:,2),fpos(:,1),'r+'); hold off;
figure(2);imshow(orgim)
hold on; scatter(fpos(:,2),fpos(:,1),'r+'); hold off;

% substract the guided backpropagation
[bpm maxidx] = max(BP,[],3);
maxidx(bpm==0) = 0;
BPS = zeros(Ny,Nx,Nz);
BPM = zeros(Ny,Nx,Nz);
BPSM = zeros(Ny,Nx,Nz);
tmp = zeros(Ny,Nx);
for jj=1:Nz;
    tmp = tmp + exp(BP(:,:,jj));
end
mask = im2bw(max(BP,[],3),0.001);
for fidx=1:Nz;
    pos = fpos(fidx,:);% [y x]
    % substract
    aa = sum(BP(:,:,setdiff([1:Nz],fidx)),3);
    bp = BP(:,:,fidx) - aa;
    bp(bp<0) = 0;
    bp = bp/max(bp(:));
    BPS(:,:,fidx) = bp;
    % max
    idx = find(maxidx~=fidx);
    bpm = BP(:,:,fidx);
    bpm(idx) = 0;
    BPM(:,:,fidx) = bpm;
    % softmax
    bpsm = exp(BP(:,:,fidx))./tmp.*im2bw(max(BP,[],3),0.001);
    bpsm(find(BP(:,:,fidx)==0)) = 0;
    BPSM(:,:,fidx) = bpsm;

    a = zeros(Ny,Nx,3);  b = zeros(Ny,Nx,3);
    a(:,:,1) = orgim + bpm.^0.5;
    a(:,:,2) = orgim - bpm.^0.5;
    a(:,:,3) = orgim - bpm.^0.5;
    b(:,:,1) = orgim + bpsm;
    b(:,:,2) = orgim - bpsm;
    b(:,:,3) = orgim - bpsm;

    if debug
        figure(3);imshow(a);hold on; scatter(pos(2),pos(1),'g+'); hold off;
        figure(4);imshow(b);hold on; scatter(pos(2),pos(1),'g+'); hold off;
    end
end

RGB = label2rgb(maxidx,'jet','black','shuffle');
figure(5);imshow(RGB)

% pre process for preconditionig
pbw = im2bw(preim,0.01);
figure(11);imshow(pbw);
hold on; scatter(fpos(:,2),fpos(:,1),'r+'); hold off;

C = bwconncomp(pbw);
L = labelmatrix(C);
Area = regionprops(L,'Area');
Area = [Area(:).Area];
inds = find(Area>10);
L = ismember(L,inds);
C = bwconncomp(L);
L = labelmatrix(C);
IDlist = [];
preimlist = zeros(size(preim));
for fidx=1:C.NumObjects;
    inds = C.PixelIdxList{fidx};
    lllist = [];
    for jj=1:Nz;
        bpm = BPM(:,:,jj);
        ll = length(find(bpm(inds)>0));
        lllist(jj) = ll;
    end
    if max(lllist)>0;
        [ll id] = max(lllist);
        IDlist(fidx) = id;
        preimlist(inds) = id;
    else
        IDlist(fidx) = -1;
    end
end
pbw2 = preimlist;
pbw2 = imfill(pbw2,'holes');
figure(11);imshow(pbw2);
hold on; scatter(fpos(:,2),fpos(:,1),'r+'); hold off;

preim2 = preim;
preim2(pbw - pbw2>0) = 0;
figure(12);imshow(preim2)

[Gy Gx] = gradient(orgim);
G = sqrt(Gx.^2 + Gy.^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gaussian fitting
% % initialize
% sigma = 9;
% FilterSize = 81;
% LL = zeros(size(orgim));
% for jj=1:size(fpos,1);
%     x = fpos(jj,2);
%     y = fpos(jj,1);
%     img_t = zeros(size(orgim));
%     img_t(y,x) = 1;
%     img_t = imgaussfilt(img_t,sigma,'FilterSize',FilterSize);
%     LL(:,:,jj) = img_t/max(img_t(:));
% end
% LL = LL/max(LL(:));
% if debug
%     figure(3); imshow(max(LL,[],3));
% end
% [Fmax Fmaxid] = max(LL,[],3);
% 
% % initialize mean and sigma
% [X1 X2] = meshgrid([1:size(orgim,1)],[1:size(orgim,2)]);
% mulist = [];
% sigmaList = [];
% FF = [];
% tmpmask = zeros(size(orgim));
% for jj=1:Nz;
%     bpmf = BPM(:,:,jj);
%     bpmb = max(BPM(:,:,setdiff([1:Nz],jj)),[],3);
%     [y x] = find(bpmf>0.0001);
%     region = [y x];
%     inds = sub2ind(size(orgim),y,x);
%     pval = bpmf(inds);
%     pval(pval<=0) = 0.0000001;
%     pval = pval(:) / sum(pval); % normalize
%     mulist(jj,:) = pval' * region;
%     Sigma = weightedcov(region, pval);
%     F1 = mvnpdf([X1(:) X2(:)],[mulist(jj,1),mulist(jj,2)],Sigma);
%     F1 = reshape(F1,size(orgim,2),size(orgim,1))';
%     F1 = F1/max(F1(:));
%     FF(:,:,jj) = F1;
%     bw = im2bw(F1,0.1);
%     tmpmask(bw>0) = 1;
% end
% [F II] = max(FF,[],3);
% if debug;
%     figure(13);imagesc(F/max(F(:)));
% end

%%%%%%%%%%%%%%%%%%%%%%
%% GraphCut
sigma=1.5;     % scale parameter in Gaussian kernel
G=fspecial('gaussian',15,sigma);
Img_smooth=conv2(orgim,G,'same');  % smooth image by Gaussiin convolution
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.

c0=2;
A_th = 40;
AL = zeros(size(orgim));
for fidx=1:Nz;
    pos = fpos(fidx,:);
    bpmf = BPM(:,:,fidx);
    BWf = im2bw(bpmf,0.0001);
    BWf = imfill(BWf,'holes');
    C = bwconncomp(BWf);
    L = labelmatrix(C);
    Area = regionprops(L,'Area');
    Area = [Area(:).Area];
    inds = find(Area >= A_th);
    BWf = ismember(L,inds);
    if debug;
        figure(6);imshow(BWf);
    end

%     tmp = exp(-rimg*10).^0.5;
%     mm = median(tmp(:));
%     tmp1 = tmp;
%     tmp1(find(tmp>=mm)) = 0.5*(tmp(find(tmp>=mm))-mm)/max(tmp(:)-mm) + 0.47;
%     tmp1(find(tmp<mm)) = 0.5*(tmp(find(tmp<mm)))/max(tmp(find(tmp<mm))) - 0.03;
    tmpf = preim2;
    tmpf(find(pbw2~=fidx)) = 0;
    tmpb = preim2;
    tmpb(find(pbw2==fidx)) = 0;
    Dc1 = 0.45*ones(size(preim2)) + tmpf - tmpb;
    Dc2 = 1 - Dc1;
    
    finds = find(BWf>0);
    Dc1(finds) = 1000000;
    Dc2(finds) = 0;
    Dc = zeros(size(Dc1));
    Dc(:,:,1) = Dc1;
    Dc(:,:,2) = Dc2;

    Sc = [0 1;1 0];
    clear gch
    gch = GraphCut('open', Dc, 50*Sc, exp(-orgim*10), exp(-orgim*10));
%     gch = GraphCut('open', Dc, 80*Sc, G.^0.5, G.^0.5);
    [gch L] = GraphCut('expand',gch);
    gch = GraphCut('close', gch);
    L = bwlabel(L);
    id = L(pos(1),pos(2));
    L = ismember(L,id);
    AL(find(L==1)) = fidx;
    if debug;
        figure(7);imshow(Dc1);
        figure(12);imshow(double(L));
    end
end
gcbw = boundarymask(AL);
rgb = imoverlay(orgim,gcbw,[1 0 0]);
rgb2 = imoverlay(RGB,gcbw,[1 0 0]);
figure(16);imshow(rgb,'InitialMagnification',67);
figure(17);imshow(rgb2,'InitialMagnification',67);

1;


