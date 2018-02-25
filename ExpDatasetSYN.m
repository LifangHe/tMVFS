classdef ExpDatasetSYN < ExpDataset
    
    methods
       function s = ExpDatasetSYN()
            s = s@ExpDataset('Synthetic', 'multi-view data');
       end
        
       function [train_data, train_label, test_data, test_label] = load(...
               varargin)
            num_samples = randi(100);
            num_view = randi(10);
            index = [];
            for i = 1 : num_view
                view_dim = randi(100);
                index = [index view_dim];
            end;
            index = cumsum(index);
            features = randn(max(index), num_samples);
            labels = randn(1, num_samples);
            labels(labels >= 0) = 1;
            labels(labels < 0) = -1;
            train_label = labels;
            train_data{1} = features;
            train_data{2} = index;
            test_data = [];
            test_label = [];
            return;
       end
    end
end