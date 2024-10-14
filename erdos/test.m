% This file runs the recSVD algorithm described in [1] on the provided example_graph.txt
%
%
% Written by Dora Erdos, Boston University, 2013
%
% 
% If using this code please cite:
% [1]: Dora Erdos, Rainer Gemulla, Evimaria Terzi "Reconstructing Graphs from Neighborhood Data", 
% IEEE International Conference on Data Mining, 2012, Brussels, Belgium, December 2012.
%
output_dir = 'results/email/'
dataset_name = 'email.txt'

% create input data to the recSVD algorithm.
M = load(dataset_name);
M = spconvert(M);
M = sparse(M);
L = M*M';
R = M'*M;

[n,~] = size(L);
[m,~] = size(R);

%run reconstruction algorithm   
[Mhat,U,SI,V] = recSVD(L,R,0,0.5,output_dir);

