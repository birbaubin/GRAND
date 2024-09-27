function[Mhat,U,SI,V] = recSVD(L,R,num_sing,thres,output_dir)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input:
% L : Squared symmetric matrix containing the left neighborhood information, can be computed as L = M*M'.
% R : Squared symmetric matrix containing the left neighborhood information.
% num_sing : Maximum number of singular values to use in SVD.
% thres : Rounding threshold to obtain binary matrix. To get the same results as in our paper use thres = 0.5
% output_dir : full path to folder where output is written
%
% output:
% Mhat : Binary matrix estimating the true bi-adjacency matrix of the graph
% U : Estimated left singular matrix of M
% SI : Estimated diagonal matrix containing the singular values of M
% V : Estimated right singular matrix of M
%
% description:
% This function runs the recSVD algorithm described in [1].
% All resulting matrices are written to file. 
%
% Written by Dora Erdos, Boston University, 2013
%
% 
% If using this code please cite:
% [1]: Dora Erdos, Rainer Gemulla, Evimaria Terzi "Reconstructing Graphs from Neighborhood Data", 
% IEEE International Conference on Data Mining, 2012, Brussels, Belgium, December 2012.
%

[n,~] = size(L);
[m,~] = size(R);



% Compute matrices U,S and V that participate in the SVD decomposition of the original M (M ~ U*S*V' with proper sign-assignment of S)
% The diagonal elements of S are all positive
[U,S,V] = my_svd(L,R,num_sing);
%write resulting SVD matrices to file
dlmwrite(strcat(output_dir,'U'),full(U),'\t');
dlmwrite(strcat(output_dir,'S'),full(S),'\t');
dlmwrite(strcat(output_dir,'V'),full(V),'\t');

% SI is the diagonal matrix containing the signed singular values in S
% where signs are computed based on our greedy heuristic
SI = greedy_signs(U,S,V,num_sing,thres);
dlmwrite(strcat(output_dir,'SI'),full(SI),'\t');

% Compute the binary aprroximation of the original matrix M matrix 
Mhat = mtx_round(U*SI*V',thres);
dlmwrite(strcat(output_dir,"M"),full(Mhat),'\t');


