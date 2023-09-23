clc
close all
clear all
State_model.Q = 3;
Measurement_model.R = 0.1;
State_model.transfer = @(x,k) 1 + sin(0.04*pi*k) + 0.5*x;
State_model.addnoise = @(x) x + sqrt(State_model.Q)* randn(1,length(x));
Measurement_model.transfer = @(x) 0.2 * x.^2;
Measurement_model.addnoise = @(y) y + sqrt(Measurement_model.R)* randn;
x = zeros(1,100);
z_real = zeros(1,100);
z_addnoise = zeros(1,100);
x(1) = 0.1;
for i = 2:100
    x(i) = State_model.transfer(x(i-1),i-1);
    x(i) = State_model.addnoise(x(i));
    z_real(i) = Measurement_model.transfer(x(i));
    z_addnoise(i) = Measurement_model.addnoise(z_real(i));
end
% figure
% plot(x)
% figure
% plot(z_addnoise)
% hold on
% plot(z_real)
RMSE.RSPF = zeros(100,100);
RMSE.PF = zeros(100,100);
RMSE.PF2 = zeros(100,100);
RMSE.PF3 = zeros(100,100);
RMSE.PF4 = zeros(100,100);

for monter = 1:100
    RSPF.X_est = zeros(1,100);
    RSPF.X_est(1) = x(1);
    RSPF.particles = x(1) * ones(1,50);
    RSPF.observation = z_addnoise;
    
    RSPF2.X_est = zeros(1,100);
    RSPF2.X_est(1) = x(1);
    RSPF2.particles = x(1) * ones(1,50);
    RSPF2.observation = z_addnoise;
    
    PF.X_est = zeros(1,100);
    PF.X_est(1) = x(1);
    PF.particles = x(1) * ones(1,50);
    PF.weights = ones(1,50)/50;
    PF.observation = z_addnoise;
    PF.resampling = 1;
    
    PF2.X_est = zeros(1,100);
    PF2.X_est(1) = x(1);
    PF2.particles = x(1) * ones(1,50);
    PF2.weights = ones(1,50)/50;
    PF2.observation = z_addnoise;
    PF2.resampling = 2;
    
    PF3.X_est = zeros(1,100);
    PF3.X_est(1) = x(1);
    PF3.particles = x(1) * ones(1,50);
    PF3.weights = ones(1,50)/50;
    PF3.observation = z_addnoise;
    PF3.resampling = 3;
    
    PF4.X_est = zeros(1,100);
    PF4.X_est(1) = x(1);
    PF4.particles = x(1) * ones(1,50);
    PF4.weights = ones(1,50)/50;
    PF4.observation = z_addnoise;
    PF4.resampling = 4;

    for i = 2:100
        
        
    %     RSPF = RSPF_version1(RSPF, State_model, Measurement_model,i);
        RSPF2 = RSPF_version2(RSPF2, State_model, Measurement_model,i);
        PF = Particle_Filter(PF, State_model, Measurement_model,i);
        PF2 = Particle_Filter(PF2, State_model, Measurement_model,i);
        PF3 = Particle_Filter(PF3, State_model, Measurement_model,i);
        PF4 = Particle_Filter(PF4, State_model, Measurement_model,i);
    
    
    %     X_estPSRF2 = RSPF_version1(X_particles.RSPF2, z_addnoise(i), State_model, Measurement_model,i-1);
    %     X_est.RSPF2(i) = X_estPSRF2.result;
    %     X_particles.RSPF2 = X_estPSRF2.particles;
    
    
    
    
    end
    RMSE.RSPF(monter,:) = RSPF2.X_est;
    RMSE.PF(monter,:) = PF.X_est;
    RMSE.PF2(monter,:) = PF2.X_est;
    RMSE.PF3(monter,:) = PF3.X_est;
    RMSE.PF4(monter,:) = PF4.X_est;

end
figure
plot(x,"Color",'b','LineStyle','-','LineWidth',1)
hold on
% plot(RSPF.X_est)
plot(RSPF2.X_est,"Color",'r','LineStyle','-','LineWidth',1)
plot(PF.X_est,"Color",'k','LineStyle','-','LineWidth',1)
plot(PF2.X_est,"Color",'c','LineStyle','-','LineWidth',1)
plot(PF3.X_est,"Color",'g','LineStyle','-','LineWidth',1)
plot(PF4.X_est,"Color",'y','LineStyle','-','LineWidth',1)
legend('Real value','Proposed algorithm','Systematic Resampling PF',...
    'Minimum variance Resampling PF',...
    'Residual Resampling PF','Branching Resampling PF')
xlabel('Time step')
ylabel('State value')
hold off
figure
% plot(x-RSPF.X_est)
hold on
plot(x-RSPF2.X_est,"Color",'r','LineStyle','-','LineWidth',1)
plot(x-PF.X_est,"Color",'k','LineStyle','-','LineWidth',1)
plot(x-PF2.X_est,"Color",'c','LineStyle','-','LineWidth',1)
plot(x-PF3.X_est,"Color",'g','LineStyle','-','LineWidth',1)
plot(x-PF4.X_est,"Color",'y','LineStyle','-','LineWidth',1)
legend('Proposed algorithm','Systematic Resampling PF',...
    'Minimum variance Resampling PF',...
    'Residual Resampling PF','Branching Resampling PF')
xlabel('Time step')
ylabel('Relative error of X')
hold off

figure
subplot(211)
plot(z_real,"Color",'b','LineStyle','-','LineWidth',1)
hold on
plot(z_addnoise,"Color",'r','LineStyle','-','LineWidth',1)
legend('The theoretical value of Observation','The real observation')
xlabel('Time step')
ylabel('Observation')
hold off
subplot(212)
plot(z_real - z_addnoise,"Color",'k','LineStyle','-','LineWidth',1)
legend('Relative error of observation')
xlabel('Time step')
ylabel('Relative error')

RMSE.RSPFCaulut = sqrt(mean((RMSE.RSPF-x).^2));
RMSE.PFCaulut = sqrt(mean((RMSE.PF-x).^2));
RMSE.PF2Caulut = sqrt(mean((RMSE.PF2-x).^2));
RMSE.PF3Caulut = sqrt(mean((RMSE.PF3-x).^2));
RMSE.PF4Caulut = sqrt(mean((RMSE.PF4-x).^2));

figure
% plot(x-RSPF.X_est)
hold on
plot(RMSE.RSPFCaulut,"Color",'r','LineStyle','-','LineWidth',1)
plot(RMSE.PFCaulut,"Color",'k','LineStyle','-','LineWidth',1)
plot(RMSE.PF2Caulut,"Color",'c','LineStyle','-','LineWidth',1)
plot(RMSE.PF3Caulut,"Color",'g','LineStyle','-','LineWidth',1)
plot(RMSE.PF4Caulut,"Color",'y','LineStyle','-','LineWidth',1)
legend('Proposed algorithm','Systematic Resampling PF',...
    'Minimum variance Resampling PF',...
    'Residual Resampling PF','Branching Resampling PF')
xlabel('Time step')
ylabel('RMSE')
hold off

