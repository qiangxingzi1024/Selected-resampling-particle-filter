%Copyright (C) 2019 Qiang xingzi. All rights reserved
%Authors:
%-->qiangxingzi@163.com
%date:2019.10.17

%branching resampling method
function [x_t2,weight2,Index_out] = Branching_Resampling(x_t1,weight1)
%x_t1:the partciles before performing the branching resampling method
%weight:the weight of particles x_t1;
%x_t2: the particles after performing the branching resampling method
%weight2:the weight of particles x_t2
%the reprentation of particles x_t1 and x_t2
%x_t1:is a d*n matrix, d reprsent the dimensions of state,n reprsent the
%number of particles
[d,N] = size(x_t1);
k=1;
for n=1:N
    if weight1(n)>0.8*(1/N) && weight1(n)<1.2*(1/N)
        weight2(1,k) = weight1(n);
        x_t2(:,k) = x_t1(:,n);
        Index_out(k) = n;
        k = k + 1;
    else
        pro = N*weight1(n) - fix(N*weight1(n));
        if rand<pro
            num_particles = fix(N*weight1(n))+1;
        else
            num_particles = fix(N*weight1(n));
        end
        weight2(1,k:k+num_particles-1) = 1/N;
        x_t2(:,k:k+num_particles-1) = x_t1(:,n) + zeros(d,num_particles);
        Index_out(k:k+num_particles-1) = n;
        k = k + num_particles;
    end
end