function[SI] = greedy_signs(U,S,V,num_sing,thres)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input:
% U : left singular matrix of M
% S : diagonal matrix containing the unsigned (positive) singular values of M
% V : right singular matrix of M
% num_sing : Maximum number of singular values to use in SVD.
% thres : Rounding threshold to obtain binary matrix. To get the same results as in our paper use thres = 0.5
% 
% output:
% SI : diagonal matrix containing the signed singular values of M
%
% description:
% Given matrices U,V and S the algorithm computes signs for the elements in the main diagonal of S
% based on the distance of matrix U*SI*V' from a binary matrix
%
% Written by Dora Erdos, Boston University, 2013
%
%


[n,~] = size(U);
[m,~] = size(V);

% If num_sing is unknown, then set it to min(n,m)
if num_sing == 0
	num_sing = min(n,m);
end

SI = zeros(size(S));
%descending order of singular values
[~,sing_order] = sort(diag(S),'descend');
sing_order = sing_order(1:num_sing);

X = zeros(n,m);
Xplus = zeros(n,m);
Xminus = zeros(n,m);
for j=1:num_sing
	i = sing_order(j);
	B = U(:,i)*S(i,i)*V(:,i)';
	% Positive or negative sign of singular value S(i,i)
	Xplus = X + B;
	Xminus = X - B;
	% Compute distance from binary mtx, when rounding is based on threshold thres
	dPlus = norm(Xplus-mtx_round(Xplus,thres),'fro');
	dMinus = norm(Xminus-mtx_round(Xminus,thres),'fro');

	% Fix the sign so that the resulting matrix X is as close to a binary matrix as possible
	if dPlus <= dMinus
		X = Xplus;
		SI(i,i) = S(i,i);
	else
		X = Xminus;
	 	SI(i,i) = -S(i,i);
	end
end
