%Copyright (C) 2019 Qiang xingzi. All rights reserved
%Authors:
%-->qiangxingzi@163.com
%date:2019.10.18


%parameters specification:
%input:
%{
1.fstate¡ú state equation with noise model
2.fstate_2¡ú The jacobian matrix of the state equation
3.x_particles¡ú the particles of last time
4.x_weights_last¡ú the particles' weights of last time
5.Pout_ekf¡ú error correlation matrix at time step k-1. Here, there are
    n matrix, where n is the particles number. the format of Pout_ekf is a
    1*n cell,in this cell, there are n matrix conrresponding with these error
    correlation matrix.
6.Q¡ú state noise matrix
7.observation¡ú observation
8.hstate¡ú observation function
9.hstate_2¡ú The jacobian matrix of the observation
10.R¡ú observation noise matrix
11.calculate_weights¡ú the equation which is used to calculate the 
    likelihood weights, prior_weights, proposal_weights based on the pdf 
    of the observation noise model, state noise model and proposal model respectively.
    the function of calculate_weights putout a 3*1 matrix.each row reprsents 
    likelihood weights, prior_weights and proposal_weights respectively.
12.resampling_method¡údifferent reampling method, you can choose 
    Systematic_Resampling,residual_Resampling or branching_resampling.
 %}
 

%output:
%{
x_estimation:the state estiamtion at time step k
%x_particles_new:the particles at current time
%x_weights_new: the weights at current time
%Pout:error correlation matrix at time step k. Here, there are
    n matrix, where n is the particles number. the format of Pout_ekf is a
    1*n cell,in this cell, there are n matrix conrresponding with these error
    correlation matrix.
%}

%the function of extended Kalman particle filter for nonlinear dynamic systems
function [x_estimation, x_particles_new, x_weights_new,Pout] = ...
    Extended_Particle_Filter(fstate,fstate_2, x_particles,x_weights_last,Pout_ekf,Q, ...
    observation, hstate, hstate_2, R, calculate_weights, resampling_method)

if nargin<12
    resampling_method ='Systematic_Resampling';
end

% initialize
[state_number,particles_number] = size(x_particles);
x_particles_next = zeros(state_number, particles_number);
x_particles_ekf = zeros(state_number, particles_number);
observation_pre = zeros(length(observation),particles_number);
x_weights = zeros(3,particles_number);
Nthr = 2/3*particles_number; %the threshold which was used to evalute if it is necessary to resampling
for i = 1 : particles_number
    [x_particles_ekf(:,i),Pout_ekf{i}] = Extended_Kalman_Filter(x_particles(:,i),fstate,fstate_2,Q,observation,hstate,hstate_2,R,Pout_ekf{i});
    x_particles_next(:,i) = x_particles_ekf(:,i) + sqrtm(Pout_ekf{i})*randn(3,1);
    observation_pre(:,i) = hstate(x_particles_next(:,i));
    x_weights(:,i) = calculate_weights(observation,observation_pre(:,i),...
        x_particles_ekf(:,i),x_particles_next(:,i),Pout_ekf{i});
end
x_weights_mix = (x_weights_last.*x_weights(1,:).*x_weights(2,:))./x_weights(3,:);
x_weights_mix = x_weights_mix/sum(x_weights_mix);
x_estimation = sum(x_particles_next.*x_weights_mix,2);


%resampling
N_eff= 1/(sum(x_weights_mix.^2)); % evaluate the necessity of resampling
if N_eff<Nthr
    if resampling_method ==1
        Index_out = Systematic_Resampling(x_weights_mix);
        x_particles_new = x_particles_next(:,Index_out);
        x_weights_new = ones(1,particles_number)/particles_number;
        Pout = Pout_ekf(Index_out);
    elseif resampling_method==2
        Index_out = Minimum_variance_Resampling(x_weights_mix);
        x_particles_new = x_particles_next(:,Index_out);
        x_weights_new = ones(1,particles_number)/particles_number;
        Pout = Pout_ekf(Index_out);
    elseif resampling_method ==3
        Index_out = residual_Resampling(x_weights_mix);
        x_particles_new = x_particles_next(:,Index_out);
        x_weights_new = ones(1,particles_number)/particles_number;
        Pout = Pout_ekf(Index_out);
    elseif resampling_method == 4
        [x_particles_new,x_weights_new,~] = Branching_Resampling(x_particles_next,x_weights_mix);
    end
else
    x_particles_new = x_particles_next;
    x_weights_new = x_weights_mix;
    Pout = Pout_ekf;
end
