clc
clear
load('C:\Users\whk13\Desktop\SRTP\SRTP\DATA\Case_16_data.mat')
load('C:\Users\whk13\Desktop\SRTP\SRTP\DATA\Case_36_data.mat')
load('test4.mat')
load('train4.mat')
% data = [Var_16_21,Var_16_32,Var_16_98,Var_16_99,Var_16_3536,Var_16_3740,Var_16_96,Var_16_100,Var_16_101,Var_16_102,Var_16_139,Var_16_140,Var_16_254,Var_16_2,Var_16_108,Var_16_52,Var_16_119,Var_16_123,Var_16_238,Var_16_242];
 data = [datatrain' datatest'];
 data =  (data');

inputdata=data(4000:5500,:)';
agru = AGRU(inputdata,1);
traintimes=4;
mse = zeros(1,traintimes);
for i=1:traintimes
    agru.AGRU_forward(inputdata);
    agru.AGRU_backward(inputdata);
    mse(1,i) = 0.5 * agru.MSE / size(inputdata,2);
end

Attenton_scores = agru.Attention;
for i=1:agru.variables_size
    for j=1:agru.variables_size
        if i==j || Attenton_scores(i,j)<=0
            Attenton_scores(i,j) = 0;
        end
    end
end

% heatmap(Attenton_scores);
heatmap(agru.Attention);
colorbar; 