function selBasis=BasisBestSelectionLocal(img,F,basisList,kernelList,M,candiBasis,ttmask,debug)

if nargin < 5, debug = 0; end

%% Inner production calculation for phase contrast image and 
[nrow,ncol]=size(img);
orgimg = img;
mask = im2bw(F,0);
img = img.*mask;
bw = im2bw(F,0.1);

[r c] = find(mask>0);
cpos = round(mean([r c]));

maxlist = [];
inner_norm = zeros(1,M);
for m=1:M
   if length(intersect(candiBasis,m)) == 0;
        inner_norm(m)=0;
        continue;
   end
   if debug
       fprintf('Calculate inner production with %sth basis\n', num2str(m));
   end
%    ang=2*pi/M*m;
   kernel = kernelList{m};
%    kernel=getKernel(kernelparas,ang,1);
   basis = basisList{m};

   resfeature=basis*img(:); %inner product of kernel and phase contrast image 
   resfeature=reshape(resfeature,nrow,ncol);
   resfeature(resfeature<0)=0;
   resfeature2 = resfeature.*ttmask;
%    resfeature2 = resfeature.*im2bw(F,0.9);
%    tmp = im2bw(F,0.99);
%    tmp = imfilter(double(tmp),fspecial('gaussian',21,3));
%    tmp = tmp/max(tmp(:));
%    tmp = tmp.*im2bw(F,0);
%    resfeature2 = resfeature.*tmp;
   
%    bw = im2bw(resfeature,0.01);
%    CC = bwconncomp(bw);
%    L = labelmatrix(CC);
%    stats = regionprops(CC,'Area');
%    Area = [stats(:).Area];
%    ids = find(Area>30);
%    L = ismember(L,ids);
%    CC = bwconncomp(L);
%    L = labelmatrix(CC);
%    dd = [];
%    for jj=1:CC.NumObjects;
%        inds = CC.PixelIdxList{jj};
%        [r c] = ind2sub(size(L),inds);
%        region = [r c];
%        d = sqrt(sum((region' - repmat(cpos',1,size(region,1))).^2));
%        dd(jj) = min(d);
%    end
%    [d dl] = min(dd); 
%    L = ismember(L,dl);
%    resfeature2 = resfeature.*L;
   
   inner_norm(m)=norm(resfeature2(:)); %norm of inner production 
   maxlist(m) = max(resfeature2(:));
 %   discrim(m)=sum(bwimg(:));
    if debug
        figure(11);
        imagesc(kernel);
        figure(10)
%         figure;
        imshow(normalize(resfeature))
%         imshow(resfeature/2.6)
        title(['Inner production of basis with phase retardation ' num2str(m) '\times 2\pi/' num2str(M) ' and original image'] )
    end
end
[val,pos]=max(inner_norm);
selBasis=pos; 