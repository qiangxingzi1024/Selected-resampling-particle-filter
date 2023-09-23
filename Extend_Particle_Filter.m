%Copyright (C) 2019 Qiang xingzi. All rights reserved
%Authors:
%-->qiangxingzi@163.com
%date:2019.09.19


%Extend particle filter
clc
close all
clear all
path(path,'.\State_model');
path(path,'.\Observation_model')
path(path,'.\resampling_methods')
path(path,'.\important_sampling_method')

%initialization
N =100; %the number of particles
D = 1;%the number of state dimensions
D_y = 1; % the number of observation dimensions
T = 500; %duration
EX = zeros(D,T);%state estimation
RX = zeros(D,T); %real state
y = zeros(D_y,T); % observation
x_t1 = zeros(D,N); %particles at time t-1
x_t2 = zeros(D,N); %particles at time t
weight_l = zeros(1,N); %likelihood weights
weight_pr = ones(1,N)/N; %prior weights



% the parametric of state_model nois and observation noise

    % state_model 
        delt1 = 3; delt2 = 2; % gamma distribution

    % observation_model
        delt3 = 0.1; %Gaussian distribution

%inital of the dynamic system
N_th = 2/3*N;
RX(:,1) = 0.1;
% x_t1 = -6 + gamrnd(delt1,delt2,D,N);
x_t1 = 0.1+randn(D,N);
EX(:,1) = mean(x_t1);

% generate the real state and the observation 
for t = 2:T
    RX (:,t) = feval('ffun',RX(:,t-1),t) + gamrnd(delt1,delt2);
    y(:,t) = feval('hfun',RX(:,t)) + delt3*randn;
end


%the procedure of standard particle filter
for t=2:T
    
    %important sampling method IS   EKFµ÷ÓÃ
    for n = 1:N  
        x_t1 (:,n) = feval('ffun',x_t1(:,n),t)+gamrnd(delt1,delt2);
    end
    %important sampling method END
    
    weight_l = calculate_likelihood_w(x_t1,y(:,t),delt3,N);
    mix_weight = (weight_l.*weight_pr)/(sum(weight_l.*weight_pr));
    
    % State estimation
    EX(:,t) = sum(x_t1.*mix_weight);
    %END
    
    N_eff= 1/(sum(mix_weight.^2)); % evaluate the necessity of resampling
    if N_eff<N_th
        
        %resampling method
        Index_out = Systematic_Resampling(mix_weight);
        x_t2 = x_t1(:,Index_out);
        weight_pr = ones(1,N)/N;
        x_t1 = x_t2;
        %resampling END
        
    else
        weight_pr = mix_weight;  %without resampling 
    end
end
figure
plot(RX,'r')
hold on 
plot(EX,'b')
figure

plot(RX-EX)