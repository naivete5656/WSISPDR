
function [basisList kernelList angleList]=makeBasis(img,kernelparas,M,debug)

if nargin < 3, debug = 0; end

%% make basis
[nrow,ncol]=size(img);

basisList = [];
kernelList = [];
angleList = [];
for m=1:M
   if debug
       fprintf('Calculate inner production with %sth basis\n', num2str(m));
   end
   ang=2*pi/M*m;
   kernel=getKernel(kernelparas,ang,0);
%    kernel=getKernel(kernelparas,ang,1);
   basis=imgfun(kernel,nrow,ncol);
   basisList{m} = basis;
   kernelList{m} = kernel;
   angleList(m) = ang;
end
1;