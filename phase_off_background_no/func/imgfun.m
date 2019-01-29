function basis=imgfun(kernel,nrows,ncols)

diameter=size(kernel,1);

radius=round((diameter-1)/2);

kernel = kernel(:);
N=nrows*ncols;
    
%build the sparse H matrix
nzidx = abs(kernel) > 0.01; %very important to save memory and speed up

inds = reshape(1:N, nrows, ncols);
inds_pad = padarray(inds,[radius radius],'symmetric'); %deal with the boundary

row_inds = repmat(1:N, sum(nzidx), 1);
col_inds = im2col(inds_pad, [diameter,diameter], 'sliding'); %slide col and then row
col_inds = col_inds(repmat(nzidx, [1,N]));
vals = repmat(kernel(nzidx), N, 1);
basis = sparse(row_inds(:), col_inds(:), vals, N, N); 