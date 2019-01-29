function z=somb(x)
%SOMB 2*j1(pi*x)/(pi*x) function.
%   SOMB(X) returns a matrix whose elements are the somb of the elements 
%   of X, i.e.
%        y = 2*j1(pi*x)/(pi*x)    if x ~= 0
%          = 1                    if x == 0
%   where x is an element of the input matrix and y is the resultant
%   output element.  

%   Author(s): J. Loomis, 6-29-1999

z=zeros(size(x));
x = abs(x);
idx=find(x);
z(idx)=2.0*besselj(1,pi*x(idx))./(pi*x(idx));
