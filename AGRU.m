classdef AGRU < handle
    properties
        % hyperparameters
        seq_length % number of sample
        variables_size % number of variables
        % Attention a(i,j)
        Attention;
        % u=ax layer
        u;
        % gru * M
        gru;
        % MSE
        MSE;
    end

    methods
        % 构造函数
        function obj = AGRU(input,P)% input: input: row_variables_size*col_seq_length
            obj.seq_length = size(input,2);
            obj.variables_size = size(input,1);
            %*********************************
            obj.Attention = ones(obj.variables_size,obj.variables_size);
%             obj.Attention = randn(obj.variables_size,obj.variables_size);
%             [attention] = obj.Attention_init(input,P);
%             obj.Attention = attention;
            %***************************************************
            obj.gru = repmat(GRU(1,obj.seq_length,obj.variables_size),obj.variables_size,1);
            for m=1:obj.variables_size
                obj.gru(m,1) = GRU(1,obj.seq_length,obj.variables_size);
            end
            obj.u = zeros(obj.variables_size,obj.variables_size,obj.seq_length);
            % MSE
            obj.MSE = 0;
        end

        function [attention] = Attention_init(obj,input,p) % first parameter must be obj
            input_=input';
            X = [];
            for i=1:p
                X = [X input_(p-i+1: end-i, :)];
            end
            Y = input_(p+1: end, :);
            [Coefficient,Residual] = OLS(X,Y);
            attention = Coefficient;

%             atten = attention;
%             for i=1:13
%                 for j=1:13
%                     if i==j || atten(i,j)<0
%                         atten(i,j) = 0;
%                     end
%                 end
%             end

        end

        function [ ] = AGRU_forward(obj,input)
            softmax_Attention = softmax(obj.Attention);
            for t=2:obj.seq_length
                for i=1:obj.variables_size
%                     for j=1:obj.variables_size
                        obj.u(i,:,t-1) = softmax_Attention(i,:).*input(:,t-1)';
%                     end
                end
%                 if t>2
                    for m=1:obj.variables_size
                        % obj.gru(m,1) = GRU(1,obj.seq_length,obj.variables_size);
                        obj.gru(m,1).gru_forward(t, obj.u(m,:,t-1)');
                    end
%                 end
            end
        end

        function [ ] = AGRU_Attention_calculate_and_update(obj,m,t,input)
           obj.Attention(m,:) = obj.Attention(m,:) - (obj.gru(m,1).dE_da .* input(:,t))' ;
        end

        function [ ] = AGRU_backward(obj,input)
            % MSE
            obj.MSE = 0;
            % get gru and atttention updated
            for m=1:obj.variables_size
                for t=obj.seq_length :-1 :2
                    obj.gru(m,1).gru_calculate_and_update_parameters(t,obj.u(m,:,t-1)',m,input(:,t));
                    obj.AGRU_Attention_calculate_and_update(m,t,input);
                end
                obj.MSE = obj.MSE + obj.gru(m,1).MSE;
            end
%             obj.Attention=mse_(obj.Attention);
        end
    end
end