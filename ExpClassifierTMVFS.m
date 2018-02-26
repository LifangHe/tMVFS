classdef ExpClassifierTMVFS < ExpClassifier
    
    properties
        para_train = '-t 4';
        perctg = 0.5;
        num_iter = 50;
    end
    
    methods
        function s = ExpClassifierTMVFS()
            s = s@ExpClassifier('tMVFS', 'classification');
        end
        
        function [outputs, pre_labels, s] = classify(...
                s, train_data, train_label, test_data)
            t = cputime;
            paraStr = s.para_train;
            index = train_data{2};
            train_data = train_data{1};
            test_data = test_data{1};
            num_train = size(train_data, 2);
            num_test = size(test_data, 2);
            num_view = length(index);
            index = [0, index];

            global w p q;
            w = cell(1, num_view);
            p = zeros(1, num_view);
            q = zeros(num_train, num_view);
            
            % initialize w
            for i = 1 : num_view
                kernel = compute_kernel(...
                    train_data, train_data, index(i : i + 1));
                model = svmtrain(...
                    train_label', [(1 : num_train)' kernel], paraStr);
                SVs = train_data(index(i) + 1 : index(i + 1), model.SVs);
                w{i} = SVs * model.sv_coef;
                p(i) = w{i}' * w{i};
                q(:, i) = train_data(index(i) + 1 : index(i + 1), :)' * w{i};
            end
            
            % burn-in w
            for cbk = 1 : s.num_iter
                for i = 1 : num_view
                    update(i, index, train_data, train_label, paraStr);
                end;
            end;
            
            % feature selection
            for i = 1 : num_view
                num_round = (index(i + 1) - index(i)) * s.perctg;
                for k = 1 : num_round
                    c = abs(w{i});
                    [~, idx] = min(c);
                    train_data(index(i) + idx, :) = [];
                    test_data(index(i) + idx, :) = [];
                    index(index >= index(i) + idx) = ...
                        index(index >= index(i) + idx) - 1;
                    update(i, index, train_data, train_label, paraStr);
                end;
            end;
            
            kernel = compute_kernel(train_data, train_data, [0, max(index)]);
            model = svmtrain(...
                train_label', [(1 : num_train)' kernel], paraStr);
            kernel = compute_kernel(test_data, train_data, [0, max(index)]);
            [pre_labels, ~, outputs] = svmpredict(...
                zeros(num_test, 1), [(1 : num_test)' kernel], model);

            s.time_train = cputime-t;
            s.time = cputime - t;  
            s.time_test = s.time - s.time_train;
            
            % save running state discription
            s.abstract = [
                s.name  '(' ...
                '-time:' num2str(s.time) ...
                '-time_train:' num2str(s.time_train) ...
                '-time_test:' num2str(s.time_test) ')'];
        end
    end
end

function kernel = compute_kernel(data_x, data_y, index)
    data_x = data_x(index(1) + 1 : index(2), :);
    data_y = data_y(index(1) + 1 : index(2), :);
    kernel = data_x' * data_y;
end

function [] = update(i, index, train_data, train_label, paraStr)
    global w p q;
    num_train = size(train_data, 2);
    p_(i) = prod(p) / p(i);
    q_(:, i) = prod(q, 2) ./ q(:, i);
    x_ = train_data(index(i) + 1 : index(i + 1), :) * ...
        diag(q_(:, i)) ./ sqrt(p_(i));
    kernel = compute_kernel(x_, x_, [0 index(i + 1) - index(i)]);
    model = svmtrain(train_label', [(1 : num_train)' kernel], paraStr);
    SVs = train_data(index(i) + 1 : index(i + 1), model.SVs);
    w{i} = SVs * (model.sv_coef .* q_(model.SVs, i)) ./ p_(i);
    p(i) = sum(w{i} .* w{i});
    q(:, i) = train_data(index(i) + 1 : index(i + 1), :)' * w{i};
end