%Systematic resampling method
function Index_out = Systematic_Resampling(weight)
N=length(weight);
Index_out = zeros(1,N);
for n=1:N
    Index_out(n)=find(rand<=cumsum(weight),1);
end