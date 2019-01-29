	
function imgInit=BackgroundRemoval(cell_data,debug)
    %crop/downsize an image or not
	im = cell_data; 

	[nrows, ncols] = size(im);
	N = nrows*ncols;

	%for flatten images
	[xx yy] = meshgrid(1:ncols, 1:nrows);
	xx = xx(:); yy = yy(:);
	X = [ones(N,1), xx, yy, xx.^2, xx.*yy, yy.^2];
    
    p = X\im(:); 		%	p = (X'*X)^(-1)*X'*im(:);    
% 	stackImg('flatten before', normalize(im)*255);
%	im = reshape(im(:)-100,[nrows,ncols]);
%     background=X*p;
	imgInit = reshape(im(:)-X*p,[nrows,ncols]);

    if debug
        figure(2); clf; 
        subplot(2,2,1); imshow(cell_data,[]);title('Orginal Image')
        subplot(2,2,2); imshow(reshape(X*p,[nrows,ncols]),[]);title('background');
        subplot(2,2,3); imshow(imgInit,[]); title('flattened im'); drawnow;
    end