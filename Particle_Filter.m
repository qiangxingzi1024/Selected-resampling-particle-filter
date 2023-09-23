%Copyright (C) 2019 Qiang xingzi. All rights reserved
%Authors:
%-->qiangxingzi@163.com
%date:2019.10.17

%parameters specification:
%input:
%fstate:state equation with noise model
%x_particles: the particles of last time
%x_weights: the particles' weights of last time
%observation: the observation of current time
%hstate:observation equation
%calculate_likelihood:the equation which is used to calculate the
%likelihood weights based on the pdf of the observation noise model
%resampling_method: different reampling method, you can choose
%Systematic_Resampling,residual_Resampling or branching_resampling.


%output:
%x_estimation:the state estiamtion at time step k
%x_particles_new:the particles at current time
%x_weights_new: the weights at current time


%the function of particle filter for nonlinear dynamic systems
% function Xk2 =  Particle_Filter(Xk1,PrNosieModel,ObNoiseModel,Sys)
function PF = Particle_Filter(PF, State_model, Measurement_model,i)

% if nargin<7
%     resampling_method =1;
% end


x_particles = PF.particles;
x_weights = PF.weights;
observation = PF.observation(i);
resampling_method = PF.resampling;



% initialize
[~,particles_number] = size(x_particles);

Nthr = 2/3*particles_number; %the threshold which was used to evalute if it is necessary to resampling

%Particles Filter code
% x_particles_next = fstate(x_particles,Sys) + mvnrnd([0,0],U,particles_number)';%
% x_particles_next = fstate(x_particles,Sys) + get_noise(PrNosieModel,particles_number)';%
x_particles_next = State_model.transfer(x_particles,i-1);
x_particles_next = State_model.addnoise(x_particles_next);

observation_reltive = Measurement_model.transfer(x_particles_next) - observation;

% x_weights_likelihood = mvnpdf(observation_reltive',0,V)'+1e-99;%
x_weights_likelihood = get_noisepdf(observation_reltive',Measurement_model.R)'+1e-99;%
x_weights_mix = x_weights_likelihood.*x_weights;
x_weights_mix = x_weights_mix/sum(x_weights_mix);
x_estimation = sum(x_particles_next.*x_weights_mix,2);


%resampling
N_eff= 1/(sum(x_weights_mix.^2)); % evaluate the necessity of resampling
if N_eff<Nthr
    if resampling_method ==1
        Index_out = Systematic_Resampling(x_weights_mix);
        x_particles_new = x_particles_next(:,Index_out);
        x_weights_new = ones(1,particles_number)/particles_number;
    elseif resampling_method==2
        Index_out = Minimum_variance_Resampling(x_weights_mix);
        x_particles_new = x_particles_next(:,Index_out);
        x_weights_new = ones(1,particles_number)/particles_number;
    elseif resampling_method ==3
        Index_out = residual_Resampling(x_weights_mix);
        x_particles_new = x_particles_next(:,Index_out);
        x_weights_new = ones(1,particles_number)/particles_number;
    elseif resampling_method == 4
        [x_particles_new,x_weights_new,~] = Branching_Resampling(x_particles_next,x_weights_mix);
    end
else
    x_particles_new = x_particles_next;
    x_weights_new = x_weights_mix;
end
% Xk2.es = x_estimation;
% Xk2.particles = x_particles_new;
% Xk2.weight = x_weights_new;
PF.X_est(i) = x_estimation;
PF.weights = x_weights_new;
PF.particles = x_particles_new;
% if Sys.i>4 && Sys.i<9
%     if particles_number == 500
%         aaa = 0.02;
%     else
%         aaa = 0.03;
%     end
%     [ff,xx]  = ksdensity(x_particles_next);
%     [ff2,xx2]  = ksdensity(x_particles_new);
%     figure(1)
%     subplot(4,2,2*(Sys.i-5)+1)
%     plot(x_particles_next,-aaa*ones(1,particles_number),'o')
%     hold on
%     plot(xx,ff)
%      if particles_number == 500
%     else
%         legend('CSPF samples with weights','PF500 particles',...
%             'PF500 fitting','PF100K particles','PF100K fitting')
%     end
%     figure(1)
%     subplot(4,2,2*(Sys.i-5)+2)
%     plot(x_particles_new,-aaa*ones(1,particles_number),'o')
%     hold on
%     plot(xx2,ff2)
% %     legend('CSPF samples with weights','PF500 particles',...
% %             'PF500 fitting','PF100K particles','PF100K fitting')
% end


%NO rsampling
% x_particles_new = x_particles_next;
% x_weights_new = x_weights_mix;
