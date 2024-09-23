classdef GRU < handle
    properties
        % hyperparameters
        hidden_size % size of hidden layer of neurons
        seq_length % number of sample
        variables_size % number of variables
        learning_rate = 0.1; % learning rate 
        % model parameters
        % Z update gate
        Wz;Uz;bz
        % R reset gate
        Wr;Ur;br
        % H_hat candidate gate
        Wh;Uh;bh
        % O output
        Wo;bo;x_hat
        % gates
        z;r;h;h_hat
        % The derivative of the MSE with respect to the activation
        dE_dh
        % the derivatives of the MSE with respect to (x(u) to) the attention score aij
        dE_da
        % MSE
        MSE;

        DEh;
    end

    methods
        % 构造函数
        function obj = GRU(hidden_size,seq_length,variables_size)
            obj.hidden_size = hidden_size;
            obj.seq_length = seq_length;
            obj.variables_size = variables_size;
            % 初始化其他参数
            % model parameters
            % Z update gate
            obj.Wz = randn(hidden_size, variables_size) * 0.01; % x_coefficient to update gate
            obj.Uz = randn(hidden_size, hidden_size) * 0.01; % h_coefficient to update gate
            obj.bz = zeros(hidden_size, 1); % update gate bias
            % R reset gate
            obj.Wr = randn(hidden_size, variables_size) * 0.01; % x_coefficient to reset gate
            obj.Ur = randn(hidden_size, hidden_size) * 0.01; % h_coefficient to reset gate
            obj.br = zeros(hidden_size, 1); % reset gate bias
            % H_hat candidate gate
            obj.Wh = randn(hidden_size, variables_size) * 0.01; % x_coefficient to candidate gate
            obj.Uh = randn(hidden_size, hidden_size) * 0.01; % h_coefficient to candidate gate
            obj.bh = zeros(hidden_size, 1); % candidate gate bias
            % O output
            obj.Wo = randn(variables_size, hidden_size) * 0.01; % h_coefficient to output
            obj.bo = zeros(variables_size, 1); % output bias
            obj.x_hat = zeros(variables_size, seq_length); % output
            % gates
            %*********************************************************
            obj.z = zeros(hidden_size, seq_length); % update gates
            obj.r = zeros(hidden_size, seq_length); % reset gates
            obj.h = zeros(hidden_size, seq_length); % hidden
            obj.h_hat = zeros(hidden_size, seq_length); % candidates
            % The derivative of the MSE with respect to the activation
            obj.dE_dh = zeros(hidden_size,seq_length);
            % the derivatives of the MSE with respect to (x(u) to) the attention score aij
            obj.dE_da = zeros(variables_size,1);
            % MSE
            obj.MSE = 0;

        end

        function [ ]= gru_forward(obj,t,input)% input: row_variables_size*col_seq_length
            % GRU gates
            obj.z(:, t) = sigmoid(obj.Wz * input + obj.Uz * obj.h(:, t - 1) + obj.bz); % update gates
            obj.r(:, t) = sigmoid(obj.Wr * input + obj.Ur * obj.h(:, t - 1) + obj.br); % reset gates
            % candidates
            obj.h_hat(:, t) = tanh(obj.Wh * input + obj.Uh * (obj.r(:, t) .* obj.h(:, t - 1)) + obj.bh);
            % new hidden state
            obj.h(:, t) = (1 - obj.z(:, t)) .* obj.h(:, t - 1) + obj.z(:, t) .* obj.h_hat(:, t);
            % output x_estimate
            obj.x_hat(:, t) = sigmoid(obj.Wo * obj.h(:, t) + obj.bo);
            % MSE
            obj.MSE = 0;
        end

        function [ ]= gru_calculate_and_update_parameters(obj,t,input,m,realx_inputdata)% input: row_variables_size*col_seq_length
            % output gate
            % The derivatives of the MSE with respect to the gru ouput
            dE_dx_hat = obj.x_hat(:,t)-realx_inputdata;
             % collect MSE
            obj.MSE = obj.MSE + dE_dx_hat(m,1)^2;
            % The derivatives of the MSE with respect to the Wo
            dE_dWo =  dE_dx_hat .* obj.x_hat(:,t) .* (1-obj.x_hat(:,t)) * obj.h(:,t)';
            % The derivatives of the MSE with respect to the bo
            dE_dbo =  dE_dx_hat .* obj.x_hat(:,t) .* (1-obj.x_hat(:,t));
            % The derivatives of the MSE with respect to the ht
            if t == obj.seq_length
                obj.dE_dh(:,t) = obj.Wo' * dE_dx_hat;
            else
                obj.dE_dh(:,t) = obj.Wo' * dE_dx_hat...
                    + obj.dE_dh(:,t+1) .* (1-obj.z(:,t+1))...
                    + obj.dE_dh(:,t+1) .* (obj.h_hat(:, t+1)- obj.h(:, t)) .* (obj.Uz' * (obj.z(:,t+1).*(1-obj.z(:,t+1))))...
                    + obj.dE_dh(:,t+1) .* (obj.Uh' * (obj.z(:,t+1) .* (1-obj.h_hat(:,t+1).^2))) .* obj.h(:,t) .* (obj.Ur' * (obj.r(:,t+1) .* (1-obj.r(:,t+1))))...
                    + obj.dE_dh(:,t+1) .* obj.z(:,t+1) .* (obj.Uh' * (1-obj.h_hat(:,t+1).^2)) .* obj.r(:,t+1);

%                 if obj.dE_dh(:,t) > (obj.dE_dh(:,obj.seq_length)+1)
%                     obj.dE_dh(:,t) = (obj.dE_dh(:,obj.seq_length)+1);
%                 end
%                 if obj.dE_dh(:,t) < (obj.dE_dh(:,obj.seq_length)-1)
%                     obj.dE_dh(:,t) = (obj.dE_dh(:,obj.seq_length)-1);
%                 end
                obj.dE_dh(:,t) = gradient_restriction(obj.hidden_size,obj.dE_dh(:,t),obj.dE_dh(:,obj.seq_length));
            end
            % update gate
            % The derivatives of the MSE with respect to the Wz
            dE_dWz = obj.dE_dh(:,t) .* (obj.h_hat(:,t)-obj.h(:,t-1)) .* (obj.z(:,t) .* (1-obj.z(:,t))) * input';
            % The derivatives of the MSE with respect to the Uz
            dE_dUz = obj.dE_dh(:,t) .* (obj.h_hat(:,t)-obj.h(:,t-1)) .* (obj.z(:,t) .* (1-obj.z(:,t))) * obj.h(:,t-1)';
            % The derivatives of the MSE with respect to the bz
            dE_dbz = obj.dE_dh(:,t) .* (obj.h_hat(:,t)-obj.h(:,t-1)) .* (obj.z(:,t) .* (1-obj.z(:,t))) ;
            % reset gate
            % The derivatives of the ht with respect to the rt
            dht_drt = (obj.Uh' * (obj.z(:,t) .* (1-obj.h_hat(:,t).^2))) .* obj.h(:,t-1);
            % The derivatives of the MSE with respect to the Wr
            dE_dWr = obj.dE_dh(:,t) .* dht_drt .* (obj.r(:,t) .* (1-obj.r(:,t))) * input';
            % The derivatives of the MSE with respect to the Ur
            dE_dUr = obj.dE_dh(:,t) .* dht_drt .* (obj.r(:,t) .* (1-obj.r(:,t))) * obj.h(:,t-1)';
            % The derivatives of the MSE with respect to the br
            dE_dbr = obj.dE_dh(:,t) .* dht_drt .* (obj.r(:,t) .* (1-obj.r(:,t))) ;
             % candidate gate
            % The derivatives of the MSE with respect to the Wh
            dE_dWh = obj.dE_dh(:,t) .* obj.z(:,t) .* (1-obj.h_hat(:,t).^2) * input';
            % The derivatives of the MSE with respect to the Uh
            dE_dUh = obj.dE_dh(:,t) .* obj.z(:,t) .* (1-obj.h_hat(:,t).^2) * (obj.r(:,t) .* obj.h(:,t-1))';
            % The derivatives of the MSE with respect to the bh
            dE_dbh = obj.dE_dh(:,t) .* obj.z(:,t) .* (1-obj.h_hat(:,t).^2);
            % W_U_b update
            % update gate
            obj.Wz = obj.Wz - obj.learning_rate .* dE_dWz;
            obj.Uz = obj.Uz - obj.learning_rate .* dE_dUz;
            obj.bz = obj.bz - obj.learning_rate .* dE_dbz;
            % reset gate
            obj.Wr = obj.Wr - obj.learning_rate .* dE_dWr;
            obj.Ur = obj.Ur - obj.learning_rate .* dE_dUr;
            obj.br = obj.br - obj.learning_rate .* dE_dbr;
           % candidate gate
            obj.Wh = obj.Wh - obj.learning_rate .* dE_dWh;
            obj.Uh = obj.Uh - obj.learning_rate .* dE_dUh;
            obj.bh = obj.bh - obj.learning_rate .* dE_dbh;
            % output gate
            obj.Wo = obj.Wo - obj.learning_rate .* dE_dWo;
            obj.bo = obj.bo - obj.learning_rate .* dE_dbo;
            % the derivatives of the MSE with re.spect to (x(u) to) the attention score aij
            obj.dE_da = obj.learning_rate .*((obj.Wz' * (obj.dE_dh(:,t) .* (obj.h_hat(:,t)-obj.h(:,t-1)) .* (obj.z(:,t) .* (1-obj.z(:,t)))))+ ...
                (obj.Wh' * (obj.dE_dh(:,t) .* (obj.z(:,t) .* (1-obj.h_hat(:,t).^2)))));
            
            obj.DEh = obj.dE_dh';
        end
    end
end
