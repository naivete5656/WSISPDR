function [ii jj] = sparse_adj_matrix(sz, r, p, st)
%
% Construct sparse adjacency matrix (provides ii and jj indices into the
% matrix)
%
% Usage:
%   [ii jj] = sparse_adj_matrix(sz, r, p, st)
%
% inputs:
%   sz - grid size (determine the number of variables n=prod(sz), and the
%        geometry)
%   r  - the radius around each point for which edges are formed
%   p  - in what p-norm to measure the r-ball, can be 1,2 or 'inf'
%   st - integer step size in making the neighborhood: st=1, full neighborhood, 
%        for r > st > 1 the neighborhood is uniformly sampled
%
% outputs
%   ii, jj - linear indices into adjacency matrix (for each pair (m,n)
%   there is also the pair (n,m))
%
% How to construct the adjacency matrix?
% >> A = sparse(ii, jj, ones(1,numel(ii)), prod(sz), prod(sz));
%
%
% Example:
% >> [ii jj] = sparse_adj_matrix([10 20], 1, inf);
% construct indices for 200x200 adjacency matrix for 8-connect graph over a
% grid of 10x20 nodes.
% To visualize the graph:
% >> [r c]=ndgrid(1:10,1:20);
% >> A = sparse(ii, jj, ones(1,numel(ii)), 200, 200);;
% >> gplot(A, [r(:) c(:)]);
%
%
%
% Copyright (c) Bagon Shai
% Department of Computer Science and Applied Mathmatics
% Wiezmann Institute of Science
% http://www.wisdom.weizmann.ac.il/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
%
% Sep. 2010
%


if nargin < 4
    st = 1;
end

% number of variables
n = prod(sz);
% number of dimensions
ndim = numel(sz);

tovec = @(x) x(:);
N=cell(ndim,1);
I=cell(ndim,1);
% construct the neighborhood
fr=floor(r);
for di=1:ndim
%     tmp = unique( round( logspace(0, log10(fr), st)-1 ) );    
    N{di}=  -fr:st:fr;
    I{di}=1:sz(di);
end
[N{1:ndim}]=ndgrid(N{:});
[I{1:ndim}]=ndgrid(I{:});
N = cellfun(tovec, N, 'UniformOutput',false);
N=[N{:}];
I = cellfun(tovec, I, 'UniformOutput',false);
I=[I{:}];

% compute N radius according to p
switch lower(p)
    case {'1','l1',1}
        R = sum(abs(N),2);
    case {'2','l2',2}
        R = sum(N.*N,2);
        r=r*r;
    case {'inf',inf}
        R = max(abs(N),[],2);
    otherwise
        error('sparse_adj_matrix:norm_type','Unknown norm p (should be either 1,2 or inf');
end
N = N(R<=r+eps,:);

% "to"-index (not linear indices)
ti = bsxfun(@plus, permute(I,[1 3 2]), permute(N, [3 1 2]));
sel = all(ti >= 1, 3) & all( bsxfun(@le, ti, permute(sz, [1 3 2])), 3);
csz = cumprod([1 sz(1:(ndim-1))]);
jj = sum( bsxfun(@times, ti-1, permute(csz, [1 3 2])), 3)+1; % convert to linear indices
ii = repmat( (1:n)', [1 size(jj,2)]);
jj = jj(sel(:));
ii = ii(sel(:));

