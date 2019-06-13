% 
%   \alpha expansion optimization for multi label non-submodular energy
%  
%   Usage: 
%     [x e] = compact_a_expand_mex(UTerm, PTerm_i, PTerm_v, [itr], [ig]);
%  
%   Inputs:
%    UTerm   - LxN matrix of unary terms (N-number of variables, L-number of labels)
%              Each term (col) is a pair [Ei(1), Ei(2), Ei(3), ...].'
%    PTerm_i - NxN sparse matrix of indices into pairwise terms 
%              Each term (col i) lists its neighbors j's.
%              Where i,j are the indices of the variables 
%  		       w_ij is index into PTerm_v (1-based index)
%    PTerm_v - (L^2)x|vi| matrix of the different col-stack matrices V. (1 based index)
%    itr     - number of iterations (optional)
%    ig      - "hot start" 1xN matrix, with initial labeling.  (optional)
%  
%   Outputs:
%  	x	  - double L vectr of final labels
%  	e	  -	[Energy Eunary Epair-wise nIter]
%           where: Energy = Eunary + Epair-wise 
%  
% 
%   
%   Input formats:
%    PTerm_i - sparse NxN matrix 
%                                  j \ i ... From ...
%                                     +----------------
%                                  .  |
%                                  .  |
%                                  .  |
%                                  T  |
%                                  o  |     w_ij
%                                  .  |
%                                  .  |
%                                  .  |
%  
%  
%    A matrix V (to be pack in col-stack into PTerm_v)
%                                l_i \ l_j ... To ...
%                                     +----------------
%                                  .  |
%                                  .  |
%                                  .  |
%                               From  |  E_{ij}(l_i,l_j) = V[l_i + l_j*L]
%                                  .  |
%                                  .  |
%                                  .  |
%  
%    For lighting energy the matrix V should be constructed as
%    >> Vh = abs( bsxfun( @minus, sL(:,1)+sL(:,2), sL(:,1)') ) + gamma * abs( bsxfun( @minus, sL(:,2), sL(:,2)' ) );
%    >> Vv = abs( bsxfun( @minus, sL(:,1)+sL(:,3), sL(:,1)') ) + gamma * abs( bsxfun( @minus, sL(:,3), sL(:,3)' ) );
%  
%  
%   mex implementation.  
%   compiling using:
%   >> mexall
%  
%
% 
% 
%   This wrapper for Matlab was written by Shai Bagon (shaibagon@gmail.com).
%   Department of Computer Science and Applied Mathmatics
%   Wiezmann Institute of Science
%   http://www.wisdom.weizmann.ac.il/
% 
%   The \alpha-expansion move is executed using QPBO construct by Valdimir
%   Kolmogorov
% 	(available from http://pub.ist.ac.at/~vnk/software.html#QPBO):
% 
%   [1] Optimizing binary MRFs via extended roof duality
%        C. Rother, V. Kolmogorov, V. Lempitsky, and M. Szummer.
%        CVPR'2007.
% 
%   [2] Efficient Approximate Energy Minimization via Graph Cuts
%        Yuri Boykov, Olga Veksler, Ramin Zabih,
%        IEEE transactions on PAMI, vol. 20, no. 12, p. 1222-1239, November
%        2001.
%  
%   [3] Matlab Wrapper for Graph Cut.
%        Shai Bagon.
%        in https://github.com/shaibagon/GCMex, December 2011.
% 
%   This software can be used only for research purposes, you should  cite ALL of
%   the aforementioned papers in any resulting publication.
%   If you wish to use this software (or the algorithms described in the
%   aforementioned paper)
%   for commercial purposes, you should be aware that there is a US patent:
% 
%       R. Zabih, Y. Boykov, O. Veksler,
%       "System and method for fast approximate energy minimization via
%       graph cuts ",
%       United Stated Patent 6,744,923, June 1, 2004
% 
% 
%   The Software is provided "as is", without warranty of any kind.
% 
% 

