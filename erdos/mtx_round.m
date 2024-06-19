function[Xround] = mtx_round(X,thres)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input:
% X : real matrix
% thres : Rounding threshold to obtain binary matrix from X. To get the same results as in our paper use thres = 0.5
% 
% output:
% Xround : binary matrix
%
% Written by Dora Erdos, Boston University, 2013
%
%

Xround = double(X>= thres * ones(size(X)));
