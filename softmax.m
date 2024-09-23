function [X] = softmax(X)
for i=1:size(X,1)
    exp_total = 0;
    for j=1:size(X,2)
        exp_total = exp_total + exp(X(i,j));
    end
    for u=1:size(X,2)
        X(i,u) = exp(X(i,u))/exp_total;
    end
end
end

