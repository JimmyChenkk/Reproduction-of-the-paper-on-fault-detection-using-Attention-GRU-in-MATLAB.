function [data] = gradient_restriction(hidden_size,data,threshold)
errormax=0.07;
    for i=1:hidden_size
        if data(i,1) > (threshold(i,1)+errormax)
            data(i,1) = (threshold(i,1)+errormax);
        end
         if data(i,1) < (threshold(i,1)-errormax)
            data(i,1) = (threshold(i,1)-errormax);
        end
    end
end

