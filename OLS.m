function [Coefficient,Residual] = OLS(X,Y) 
% 应用最小二乘法(OLS)进行多元线性回归 Y = X*Coefficient+Residual
% X成为数据矩阵，每一行是一个样本个体的全部数据，每一列是一个解释变量的全部数据。
    Coefficient = inv(X'*X)*X'*Y; % 线性回归系数
    Residual = Y-X*Coefficient; % 残差
end