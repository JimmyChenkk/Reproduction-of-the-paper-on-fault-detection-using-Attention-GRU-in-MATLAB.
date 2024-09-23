function [E] = MSE(xt,xt_hat)
E=0;
for i=1:size(xt,1)
    for j=2:size(xt,2)+1
           E=E+( xt(i,j)-xt_hat(i,j) )^2;
    end
end
E=0.5*E;
end

