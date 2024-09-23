function [data] = mse_(data)
vars=size(data,1);
if vars==13
    data=zeros(vars,vars);
    data( 13 , 12 )   = 1.0  +  0.17  * randn;
    data( 1  , 10 )   = 0.5  +  0.17  * randn;
    data( 10 , 1 )    = 0.5  +  0.17  * randn;
    data( 10 , 11 )   = 0.4  +  0.17  * randn;
    data( 11 , 5 )    = 1.1  +  0.17  * randn;
    data( 5  , 8 )    = 1.2  +  0.17  * randn;
    data( 5  , 9 )    = 1.3  +  0.17  * randn;
    data( 5  , 3 )    = 1.5  +  0.17  * randn;
    data( 5  , 4 )    = 1.5  +  0.17  * randn;
    data( 3  , 5 )    = 0.7  +  0.17  * randn;
    data( 4  , 5 )    = 0.7  +  0.17  * randn;
    data( 3  , 8 )    = 0.7  +  0.17  * randn;
    data( 3  , 9 )    = 0.8  +  0.17  * randn;
    data( 4  , 8 )    = 0.7  +  0.17  * randn;
    data( 4  , 9 )    = 0.8  +  0.17  * randn;
    data( 2  , 7 )    = 0.7  +  0.17  * randn;
    data( 6  , 7 )    = 0.8  +  0.17  * randn;
end
end

