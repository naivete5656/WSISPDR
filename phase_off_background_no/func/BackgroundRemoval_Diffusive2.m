	
function imgInit=BackgroundRemoval_Diffusive(cell_data,maskC,debug)
    %crop/downsize an image or not
	im = cell_data; 

	[nrows, ncols] = size(im);
	N = nrows*ncols;

% 	%for flatten images
% 	[xx yy] = meshgrid(1:ncols, 1:nrows);
% 	xx = xx(:); yy = yy(:);
% 	X = [ones(N,1), xx, yy, xx.^2, xx.*yy, yy.^2];
%     
%     p = X\im(:); 		%	p = (X'*X)^(-1)*X'*im(:);    
% % 	stackImg('flatten before', normalize(im)*255);
% %	im = reshape(im(:)-100,[nrows,ncols]);
% 	imgInit = reshape(im(:)-X*p,[nrows,ncols]);
%       
%     tmp = double(imgInit);
%     tmp(tmp<0) = 0;
%     [FX FY] = gradient(imgInit);
%     F = sqrt(FX.^2+FY.^2);
%     tmp2 = imfilter(double(tmp),fspecial('gaussian',101,17),'replicate').*maskC;
%     imgInit = imgInit - 2*tmp2;
    
%     tmp2 = imfilter(double(tmp),fspecial('gaussian',101,23)).*maskC;
%     imgInit = imgInit - 3*tmp2;

    % flatten by diffusive bias
    tmp = im;
%     tmp3 = imfilter(double(im),fspecial('gaussian',31,3),'replicate');
%     tmp3 = (tmp3-min(tmp3(:)))/(max(tmp3(:))-min(tmp3(:)));
%     tmp = im - tmp3.*median(im(:));
%     tmp3 = imfilter(double(im),fspecial('gaussian',101,15),'replicate').*maskC;
%     tmp = im - 2*tmp3;
    tmp2 = imfilter(double(tmp),fspecial('gaussian',301,50),'replicate');
    tmp2 = (tmp2-min(tmp2(:)))/(max(tmp2(:))-min(tmp2(:)));
    tmp = im - tmp2.*median(im(:));
    
    %for flatten images
	[xx yy] = meshgrid(1:ncols, 1:nrows);
	xx = xx(:); yy = yy(:);
	X = [ones(N,1), xx, yy, xx.^2, xx.*yy, yy.^2];
    
    p = X\tmp(:); 		%	p = (X'*X)^(-1)*X'*im(:);   
	imgInit = reshape(tmp(:)-X*p,[nrows,ncols]);
    imgInit = imgInit - median(imgInit(:));

    if debug
        figure(2); clf; 
        subplot(2,2,1); imshow(cell_data,[]);title('Orginal Image')
        subplot(2,2,2); imshow(reshape(X*p,[nrows,ncols]),[]);title('background');
        subplot(2,2,3); imshow(imgInit,[]); title('flattened im'); drawnow;
    end