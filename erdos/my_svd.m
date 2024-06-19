function[U,S,V] = my_svd(L,R,num_sing)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input:
% L : Squared symmetric matrix containing the left neighborhood information, can be computed as L = M*M'.
% R : Squared symmetric matrix containing the left neighborhood information.
% num_sing : Maximum number of singular values to use in SVD.
%
% output:
% U : Left singular matrix of M
% S : Diagonal matrix containing the (unsigned) singular values of M
% V : Right singular matrix of M
%
% Written by Dora Erdos, Boston University, 2013
%
%

[n,~] = size(L);
[m,~] = size(R);

if num_sing == 0
   num_sing = min(n,m);
end

[U,SU] = eigs(L,num_sing);
[V,SV] = eigs(R,num_sing);

SU = max(SU,0);
SV = max(SV,0);
S = (SU+SV)/2;
S = sqrt(S);
