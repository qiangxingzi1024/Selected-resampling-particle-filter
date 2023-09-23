function [ p ] = likelihood( Z_est, Z_mea, sig_meas )


    var = sig_meas;
    p =  exp( -(Z_mea - Z_est)^2/(2*var) );
%     p = (1/sqrt(2*pi*var)) * exp( -(Z_mea - Z_est)^2/(2*var) );

end

