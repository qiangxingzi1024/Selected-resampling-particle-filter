function P = get_noisepdf(X_data,NosieModel)
% if NosieModel.style ==1 %Gaussian distribution
    P = exp(-0.5.*sum(X_data*inv(NosieModel).*X_data,2)) + 10e-99; %get an n-by-1 vector.
% elseif NosieModel.style == 2   % Student-t distribution
%     P = mvtpdf(X_data,NosieModel.U,NosieModel.df); 
% end