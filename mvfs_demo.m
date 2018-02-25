% Reference:
% Bokai Cao, Lifang He, Xiangnan Kong, Philip S. Yu, Zhifeng Hao and 
% Ann B. Ragin. Tensor-based Multi-view Feature Selection with Applications
% to Brain Diseases. In ICDM 2015.
%
% Dependency:
% Chih-Chung Chang and Chih-Jen Lin. 
% LIBSVM: A Library for Support Vector Machines.
% In ACM Transactions on Intelligent Systems and Technology 2011.
% Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

clear
clc

addpath(genpath('./libsvm-3.22/matlab'));

dataset = ExpDatasetSYN();
[train_data, train_label] = dataset.load();
test_data = train_data; % for demo purpose

classifier = ExpClassifierMVFS();

[outputs, pre_labels, model] = classifier.classify(...
    train_data, train_label, test_data);
