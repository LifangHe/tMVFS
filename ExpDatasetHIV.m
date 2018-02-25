classdef ExpDatasetHIV < ExpDataset
    
    methods
       function s = ExpDatasetHIV()
            s = s@ExpDataset('HIV', 'multi-view data');
       end
        
       function [train_data, train_label, test_data, test_label] = load(...
               varargin)
            load('HIV_multi.mat');
            train_label = labels';
            train_data{1} = features';
            train_data{2} = index;
            test_data = [];
            test_label = [];
            return;
       end
    end
end