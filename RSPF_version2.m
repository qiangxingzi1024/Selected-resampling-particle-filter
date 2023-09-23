


function [RSPF] = RSPF_version2(RSPF, State_model, Measurement_model,k)


    X_particles = RSPF.particles;
    Observation = RSPF.observation(k);
    [rown,columnn] = size(X_particles);
    X_next_particles = zeros(rown,1);
    i = 1;
    while length(X_next_particles(1,:))~=columnn+1
        Particle = State_model.transfer(X_particles(:,i),k-1);
        Particle = State_model.addnoise(Particle);
        Observation_pre = Measurement_model.transfer(Particle);
        weight = likelihood(Observation_pre, Observation, Measurement_model.R);
        if rand<weight
            X_next_particles = [X_next_particles, Particle];
%             length(X_next_particles(1,:))
        end
        i = i + 1;
        if i==columnn+1
            i=1;
        end
    end
    RSPF.particles = X_next_particles(:,2:end);
    RSPF.X_est(k) = mean(X_next_particles,2);
end
