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

% fpos = [];
% C= bwconncomp(maskF);
% stats = regionprops(C,'centroid');
% for ii=1:C.NumObjects;
%     fpos(ii,:) = round(stats(ii).Centroid);
% end

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

    figure(3);imshow(a);hold on; scatter(pos(2),pos(1),'g+'); hold off;
    figure(4);imshow(b);hold on; scatter(pos(2),pos(1),'g+'); hold off;

%     % voting
%     bpbw = im2bw(bpm,0.01);
%     [y x] = find(bpm>0.01);
%     bpv = zeros(Ny,Nx);
%     for ii=1:length(y);
%         vpos = [y(ii) x(ii)];
%         intensity = bpm(y(ii),x(ii));
%         v = pos - vpos;
%         if norm(v)==0; continue; end
%         ll = [0:0.5:ceil(norm(v))]';
%         vlist = unique(round(vpos + repmat(v/norm(v),length(ll),1).*ll),'rows');
%         inds = sub2ind(size(bpv),vlist(:,1),vlist(:,2));
%         bpv(inds) = intensity;
%     end

end

RGB = label2rgb(maxidx,'jet','black','shuffle');
figure(5);imshow(RGB)

% pre process for preconditionig
% pbw = im2bw(preim,0.2);
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
% for fidx=1:Nz;
%     pos = fpos(fidx,:);
%     bpm = BPM(:,:,fidx);
%     inds = find(bpm>0.25);
%     id = L(inds);
%     IDlist = union(IDlist,id);
%     ids = setdiff(id,0);
%     tmppre = ismember(L,ids);
%     tmppre = imfill(tmppre,'holes');
%     preimlist(:,:,fidx) = tmppre;
% end
% IDlist = setdiff(IDlist,0);
% pbw2 = ismember(L,IDlist);
% pbw2 = imfill(pbw2,'holes');
% figure(11);imshow(pbw2);
% hold on; scatter(fpos(:,2),fpos(:,1),'r+'); hold off;

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

%%%%%%%%%%%%%%%%%%%%%%
%% 
% BPM2 = zeros(size(orgim));
% for fidx=1:Nz;
%     pos = fpos(fidx,:);
%     bpmf = BPM(:,:,fidx);
%     BWf = im2bw(bpmf,0.0001);
%     BWf = imfill(BWf,'holes');
%     C = bwconncomp(BWf);
%     L = labelmatrix(C);
%     Area = regionprops(L,'Area');
%     Area = [Area(:).Area]
%     inds = find(Area >= A_th);
%     BWf = ismember(L,inds);
%     figure(6);imshow(BWf);
%     hold on; scatter(pos(2),pos(1),'r+'); hold off;
%     id = L(pos(1),pos(2));
%     bwc = ismember(L,id);
% 
%     % voting
%     bpbw = im2bw(bpm,0.01);
%     [y x] = find(bpm>0.01);
%     bpv = zeros(Ny,Nx);
%     for ii=1:length(y);
%         vpos = [y(ii) x(ii)];
%         intensity = bpm(y(ii),x(ii));
%         v = pos - vpos;
%         if norm(v)==0; continue; end
%         ll = [0:0.5:ceil(norm(v))]';
%         vlist = unique(round(vpos + repmat(v/norm(v),length(ll),1).*ll),'rows');
%         inds = sub2ind(size(bpv),vlist(:,1),vlist(:,2));
%         bpv(inds) = intensity;
%     end
% end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gaussian fitting
% initialize
sigma = 9;
FilterSize = 81;
LL = zeros(size(orgim));
for jj=1:size(fpos,1);
    x = fpos(jj,2);
    y = fpos(jj,1);
    img_t = zeros(size(orgim));
    img_t(y,x) = 1;
    img_t = imgaussfilt(img_t,sigma,'FilterSize',FilterSize);
    LL(:,:,jj) = img_t/max(img_t(:));
end
LL = LL/max(LL(:));
if debug
    figure(3); imshow(max(LL,[],3));
end
[Fmax Fmaxid] = max(LL,[],3);

% initialize mean and sigma
[X1 X2] = meshgrid([1:size(orgim,1)],[1:size(orgim,2)]);
mulist = [];
sigmaList = [];
FF = [];
tmpmask = zeros(size(orgim));
for jj=1:Nz;
    bpmf = BPM(:,:,jj);
    bpmb = max(BPM(:,:,setdiff([1:Nz],jj)),[],3);
    [y x] = find(bpmf>0.0001);
    region = [y x];
    inds = sub2ind(size(orgim),y,x);
    pval = bpmf(inds);
    pval(pval<=0) = 0.0000001;
    pval = pval(:) / sum(pval); % normalize
    mulist(jj,:) = pval' * region;
    Sigma = weightedcov(region, pval);
    F1 = mvnpdf([X1(:) X2(:)],[mulist(jj,1),mulist(jj,2)],Sigma);
    F1 = reshape(F1,size(orgim,2),size(orgim,1))';
    F1 = F1/max(F1(:));
    FF(:,:,jj) = F1;
    bw = im2bw(F1,0.1);
    tmpmask(bw>0) = 1;
end
[F II] = max(FF,[],3);
if debug;
    figure(13);imagesc(F/max(F(:)));
end

%%%%%%%%%%%%%%%%%%%%%%
%% super pixel
% [SP,NumLabels] = superpixels(orgim,2000,'Compactness',8);
% figure(14)
% tmpbw = boundarymask(SP);
% rgb = imoverlay(orgim,tmpbw,[1 0 0]);
% imshow(rgb,'InitialMagnification',67);
% 
% pbw3 = im2bw(preim,0.2).*pbw2;
% figure(15);imshow(pbw3)
% 
% inds = SP(pbw2>0);
% IDs = unique(inds);
% pIDs = [];
% for jj=1:length(IDs);
%     id = IDs(jj);
%     nf = length(find(inds==id));
%     nt = length(find(SP==id));
%     if nf/nt > 0.5;
%         pIDs(end+1) = id;
%     end
% end
% bw = ismember(SP,pIDs);
% figure(7);imshow(bw);
% 
% 
% A_th = 10;
% SPL = zeros(size(orgim));
% for fidx=1:Nz;
%     bpmf = BPM(:,:,fidx);
%     BWf = im2bw(bpmf,0.0001);
%     BWf = imfill(BWf,'holes');
%     C = bwconncomp(BWf);
%     L = labelmatrix(C);
%     Area = regionprops(L,'Area');
%     Area = [Area(:).Area]
%     inds = find(Area >= A_th);
%     BWf = ismember(L,inds);
%     rgb = imoverlay(BWf,tmpbw,[1 0 0]);
%     figure(6);imshow(rgb,'InitialMagnification',67);
% 
%     inds = SP(BWf>0);
%     IDs = unique(inds);
%     pIDs = [];
%     for jj=1:length(IDs);
%         id = IDs(jj);
%         nf = length(find(inds==id));
%         nt = length(find(SP==id));
%         if nf/nt > 0.05;
%             pIDs(end+1) = id;
%         end
%     end
%     bw = ismember(SP,pIDs);
%     figure(7);imshow(bw);
%     bw2 = imerode(bw,strel('disk',1));
%     bw3 = (bw - bw2).*BWf;
%     inds = SP(bw2>0);
%     IDs2 = unique(inds);
%     pIDs2 = [];
%     for jj=1:length(IDs2);
%         id = IDs2(jj);
%         nf = length(find(inds==id));
%         nt = length(find(SP==id));
%         if nf/nt > 0.8;
%             pIDs2(end+1) = id;
%         end
%     end
%     bw4 = ismember(SP,pIDs2);
%     [LLL] = bwlabel(bw4);
%     aaa = regionprops(LLL,'Area');
%     aaa = [aaa(:).Area];
%     [mmm inds] = max(aaa);
%     bw4 = ismember(LLL,inds);
%     tmpbw4 = boundarymask(bw4);
%     rgb = imoverlay(orgim,tmpbw4,[1 0 0]);
%     figure(7);imshow(rgb,'InitialMagnification',67);
%     SPL(bw4>0) = fidx;
% end
%%%%%%%%%%%%%%%%%%%%%%
%% Level set without precondition
%% parameter setting
timestep=5;  % time step
mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
iter_inner=5;
iter_outer=40;
lambda=4; % coefficient of the weighted length term L(phi)
alfa=0;  % coefficient of the weighted area term A(phi)
epsilon=1.5; % papramater that specifies the width of the DiracDelta function
potentialFunction = 'double-well'

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
    Area = [Area(:).Area]
    inds = find(Area >= A_th);
    BWf = ismember(L,inds);
    figure(6);imshow(BWf);

    L = bwlabel(pbw2);
    id = setdiff(L(find(BWf>0)),0);
    L = ismember(L,id);
    tmppre = preim2.*L;
    flags = L(sub2ind(size(orgim),fpos(:,1),fpos(:,2)));
    idlist = find(flags>0);
    idlist = setdiff(idlist,fidx);
    LLtmp = zeros([Ny, Nx,length(idlist)+1]);
    LLtmp(:,:,1) = LL(:,:,fidx);
    for jj=1:length(idlist);
        LLtmp(:,:,jj+1) = LL(:,:,idlist(jj));
    end
    [Fmax Fmaxid] = max(LLtmp,[],3);
    tmppre(Fmaxid>1) = 0;
    L = bwlabel(im2bw(tmppre,0));
    id = setdiff(L(find(BWf>0)),0);
    L = ismember(L,id);
    tmppre = tmppre.*L;

    prior = bpmf.^0.1.*round(BWf);
%     prior(prior==0) = tmppre(prior==0).^0.1;
%     tmpbw = BWf + tmppre;
    tmpbw = BWf;
    [y x] = find(tmpbw>=0);
    ty = max(1,min(y)-10); by = min(size(orgim,1),max(y)+10);
    lx = max(1,min(x)-10); rx = min(size(orgim,1),max(x)+10);
    tmp = bwconvhull(tmpbw);
    tmp2 = zeros(size(orgim)); tmp2(ty:by,lx:rx) = 1;
    inds = find(tmp>0);
    initialLSF=c0*ones(size(orgim));
    initialLSF(inds)=-c0;  
    phi=initialLSF;
    figure(15);
    imagesc(prior,[0, 1]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');

    mask = zeros(size(orgim));
    mask(inds) = 1;
    phi = activecontour((im2bw(prior,0))*255,mask,100,'Chan-Vese','SmoothFactor',2.5);
    L = bwlabel(phi);
    bpm = BPM(:,:,fidx);
    inds = find(bpm>0.25);
    ids = L(inds);
    [counts,centers] = hist(ids,[0:Nz]);
    [mm id] = max(counts);
    id = id - 1;
    phi = ismember(L,id);

    phibw = boundarymask(phi);
    rgb = imoverlay(orgim,phibw,[1 0 0]);
    rgb2 = imoverlay((im2bw(prior,0)),phibw,[1 0 0]);
    figure(16);imshow(rgb,'InitialMagnification',67);
    figure(17);imshow(rgb2,'InitialMagnification',67);
    AL(phi>0) = fidx;

%     % start level set evolution
%     for n=1:iter_outer
%         phi = drlse_edge(phi, bpmf, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
% %         phi = drlse_edge(phi, (1-bpmf.^0.5).*BWf, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
%         if mod(n,2)==0
%             figure(15);
%             imagesc( bpmf.^0.5,[0, 1]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
%             figure(16);
%             imagesc( orgim,[0, 1]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
%         end
%     end
% 
%     % refine the zero level contour by further level set evolution with alfa=0
%     alfa=0;
%     iter_refine = 10;
%     phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);

%     finalLSF=phi;
%     figure(15);
%     imagesc(orgim,[0, 1]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
%     hold on;  contour(phi, [0,0], 'r');
%     str=['Final zero level contour, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
%     title(str);
% 
%     pause(15);
%     figure;
%     mesh(-finalLSF); % for a better view, the LSF is displayed upside down
%     hold on;  contour(phi, [0,0], 'r','LineWidth',2);
%     str=['Final level set function, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
%     title(str);
%     axis on;

    
end
phibw = boundarymask(AL);
rgb = imoverlay(orgim,phibw,[1 0 0]);
rgb2 = imoverlay(RGB,phibw,[1 0 0]);
figure(16);imshow(rgb,'InitialMagnification',67);
figure(17);imshow(rgb2,'InitialMagnification',67);


%%%%%%%%%%%%%%%%%%%%%%
%% level set
%% parameter setting
% timestep=5;  % time step
% mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
% iter_inner=5;
% iter_outer=40;
% lambda=4; % coefficient of the weighted length term L(phi)
% alfa=0;  % coefficient of the weighted area term A(phi)
% epsilon=1.5; % papramater that specifies the width of the DiracDelta function
% potentialFunction = 'double-well'
% 
% sigma=1.5;     % scale parameter in Gaussian kernel
% G=fspecial('gaussian',15,sigma);
% Img_smooth=conv2(orgim,G,'same');  % smooth image by Gaussiin convolution
% [Ix,Iy]=gradient(Img_smooth);
% f=Ix.^2+Iy.^2;
% g=1./(1+f);  % edge indicator function.
% 
% c0=2;
% A_th = 40;
% AL = zeros(size(orgim));
% for fidx=1:Nz;
%     pos = fpos(fidx,:);
%     bpmf = BPM(:,:,fidx);
%     BWf = im2bw(bpmf,0.0001);
%     BWf = imfill(BWf,'holes');
%     C = bwconncomp(BWf);
%     L = labelmatrix(C);
%     Area = regionprops(L,'Area');
%     Area = [Area(:).Area]
%     inds = find(Area >= A_th);
%     BWf = ismember(L,inds);
%     figure(6);imshow(BWf);
% 
%     L = bwlabel(pbw2);
%     id = setdiff(L(find(BWf>0)),0);
%     L = ismember(L,id);
%     tmppre = preim2.*L;
%     flags = L(sub2ind(size(orgim),fpos(:,1),fpos(:,2)));
%     idlist = find(flags>0);
%     idlist = setdiff(idlist,fidx);
%     LLtmp = zeros([Ny, Nx,length(idlist)+1]);
%     LLtmp(:,:,1) = LL(:,:,fidx);
%     for jj=1:length(idlist);
%         LLtmp(:,:,jj+1) = LL(:,:,idlist(jj));
%     end
%     [Fmax Fmaxid] = max(LLtmp,[],3);
%     tmppre(Fmaxid>1) = 0;
%     L = bwlabel(im2bw(tmppre,0));
%     id = setdiff(L(find(BWf>0)),0);
%     L = ismember(L,id);
%     tmppre = tmppre.*L;
% 
%     prior = bpmf.^0.1.*round(BWf);
%     prior(prior==0) = tmppre(prior==0).^0.1;
%     tmpbw = BWf + tmppre;
%     [y x] = find(tmpbw>=0);
%     ty = max(1,min(y)-10); by = min(size(orgim,1),max(y)+10);
%     lx = max(1,min(x)-10); rx = min(size(orgim,1),max(x)+10);
%     tmp = bwconvhull(tmpbw);
%     tmp2 = zeros(size(orgim)); tmp2(ty:by,lx:rx) = 1;
%     inds = find(tmp>0);
%     initialLSF=c0*ones(size(orgim));
%     initialLSF(inds)=-c0;  
%     phi=initialLSF;
%     figure(15);
%     imagesc(prior,[0, 1]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
% 
%     mask = zeros(size(orgim));
%     mask(inds) = 1;
%     phi = activecontour((im2bw(prior,0))*255,mask,100,'Chan-Vese','SmoothFactor',2.5);
%     L = bwlabel(phi);
%     id = L(pos(1),pos(2));
%     phi = ismember(L,id);
% 
%     phibw = boundarymask(phi);
%     rgb = imoverlay(orgim,phibw,[1 0 0]);
%     rgb2 = imoverlay((im2bw(prior,0)),phibw,[1 0 0]);
%     figure(16);imshow(rgb,'InitialMagnification',67);
%     figure(17);imshow(rgb2,'InitialMagnification',67);
%     AL(phi>0) = fidx;
% 
% %     % start level set evolution
% %     for n=1:iter_outer
% %         phi = drlse_edge(phi, bpmf, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
% % %         phi = drlse_edge(phi, (1-bpmf.^0.5).*BWf, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
% %         if mod(n,2)==0
% %             figure(15);
% %             imagesc( bpmf.^0.5,[0, 1]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
% %             figure(16);
% %             imagesc( orgim,[0, 1]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
% %         end
% %     end
% % 
% %     % refine the zero level contour by further level set evolution with alfa=0
% %     alfa=0;
% %     iter_refine = 10;
% %     phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
% 
% %     finalLSF=phi;
% %     figure(15);
% %     imagesc(orgim,[0, 1]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
% %     hold on;  contour(phi, [0,0], 'r');
% %     str=['Final zero level contour, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
% %     title(str);
% % 
% %     pause(15);
% %     figure;
% %     mesh(-finalLSF); % for a better view, the LSF is displayed upside down
% %     hold on;  contour(phi, [0,0], 'r','LineWidth',2);
% %     str=['Final level set function, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
% %     title(str);
% %     axis on;
% 
%     
% end
% phibw = boundarymask(AL);
% rgb = imoverlay(orgim,phibw,[1 0 0]);
% rgb2 = imoverlay(RGB,phibw,[1 0 0]);
% figure(16);imshow(rgb,'InitialMagnification',67);
% figure(17);imshow(rgb2,'InitialMagnification',67);

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
    Area = [Area(:).Area]
    inds = find(Area >= A_th);
    BWf = ismember(L,inds);
    figure(6);imshow(BWf);

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
    figure(7);imshow(Dc1);

    Sc = [0 1;1 0];
    clear gch
    gch = GraphCut('open', Dc, 50*Sc, exp(-orgim*10), exp(-orgim*10));
%     gch = GraphCut('open', Dc, 80*Sc, G.^0.5, G.^0.5);
    [gch L] = GraphCut('expand',gch);
    gch = GraphCut('close', gch);
    L = bwlabel(L);
    id = L(pos(1),pos(2));
    L = ismember(L,id);
    figure(12);imshow(double(L));
    AL(find(L==1)) = fidx;
end
gcbw = boundarymask(AL);
rgb = imoverlay(orgim,gcbw,[1 0 0]);
rgb2 = imoverlay(RGB,gcbw,[1 0 0]);
figure(16);imshow(rgb,'InitialMagnification',67);
figure(17);imshow(rgb2,'InitialMagnification',67);



%%%%%%%%%%%%%%%%%%%%%%
%% 
A_th = 10;
bpmf = max(BPM,[],3);
BWf = im2bw(bpmf,0.0001);
BWf = imdilate(BWf,strel('disk',1));
BWf = imfill(BWf,'holes');
BWf = imerode(BWf,strel('disk',1));
BWf = imdilate(BWf,strel('disk',1));
BWf = imerode(BWf,strel('disk',1));
C = bwconncomp(BWf);
L = labelmatrix(C);
Area = regionprops(L,'Area');
Area = [Area(:).Area];
inds = find(Area >= A_th);
BWf = ismember(L,inds);
figure(13);imshow(BWf)

mm = median(orgim(:));
gimg = BackgroundRemoval_Diffusive2(orgim,ones(size(orgim)),debug);
rimg = gimg + mm; rimg(rimg>1) = 1; rimg(rimg<0) = 0;
gimg(gimg<0) = 0;
G = zeros(Ny,Nx);
A_th = 10;
for fidx=1:Nz;
    bpmf = BPM(:,:,fidx);
    bpmb = max(BPM(:,:,setdiff([1:Nz],fidx)),[],3);
    BWf = im2bw(bpmf,0.0001);
    BWb = im2bw(bpmb,0.0001);
    BWf = imfill(BWf,'holes');
    BWb = imfill(BWb,'holes');
%     BW = imdilate(BW,strel('disk',1));
%     BW = imerode(BW,strel('disk',1));
    C = bwconncomp(BWf);
    L = labelmatrix(C);
    Area = regionprops(L,'Area');
    Area = [Area(:).Area]
    inds = find(Area >= A_th);
    BWf = ismember(L,inds);
    figure(6);imshow(BWf);
    C = bwconncomp(BWb);
    L = labelmatrix(C);
    Area = regionprops(L,'Area');
    Area = [Area(:).Area]
    inds = find(Area >= A_th);
    BWb = ismember(L,inds);

    tmp = exp(-rimg*10).^0.5;
    mm = median(tmp(:));
    tmp1 = tmp;
    tmp1(find(tmp>=mm)) = 0.5*(tmp(find(tmp>=mm))-mm)/max(tmp(:)-mm) + 0.47;
    tmp1(find(tmp<mm)) = 0.5*(tmp(find(tmp<mm)))/max(tmp(find(tmp<mm))) - 0.03;
    Dc1 = tmp1 + preim2;
    Dc2 = 1 - Dc1;
    
    finds = find(BWf>0);
    binds = find(BWb>0);
    Dc1(finds) = 1000000;
    Dc2(finds) = 0;
    Dc2(binds) = 1000000;
    Dc1(binds) = 0;
    Dc = zeros(size(Dc1));
    Dc(:,:,1) = Dc1;
    Dc(:,:,2) = Dc2;
    figure(7);imshow(Dc1);

    Sc = [0 1;1 0];
    clear gch
    gch = GraphCut('open', Dc, 50*Sc, exp(-orgim*10), exp(-orgim*10));
%     gch = GraphCut('open', Dc, 80*Sc, G.^0.5, G.^0.5);
    [gch L] = GraphCut('expand',gch);
    gch = GraphCut('close', gch);
    figure(12);imshow(double(L));
    G(find(L==1)) = fidx;
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 

E = entropyfilt(bpm,true(5));
BW = im2bw(E,0.1);
BW = imerode(BW,strel('disk',2));
BW = imfill(BW,'holes');
figure(13);imshow(BW);


