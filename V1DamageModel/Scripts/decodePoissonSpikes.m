% ---------------------------------------------------------------------
% Function to decode stimulus orientation from spike data using a Bayesian approach.
% This function performs decoding based on the neuron's tuning curves and 
% the observed spike counts, estimating the most likely stimulus orientation.

% INPUTS:
% - spikes:            [nNeurons x nTimeBins] Binary matrix containing the spike responses for all neurons over time.
% - neuronsPrefOrientation: [1 x nNeurons] Preferred orientations of the neurons (in radians), determining their tuning.
% - params:            Structure containing several parameters related to the time bins, number of neurons, and stimulus duration.
%   - params.timeBins: [1 x nTimeBins] Time bin vector representing the time points for spike recording.
%   - params.nNeurons: Scalar representing the total number of neurons.
%   - params.stimDuration: Scalar representing the total duration of stimulus presentation (in seconds).
% - tuningParams:      Structure containing tuning curve parameters for the population of neurons (used for computing firing rates based on orientation).
%   - tuningParams: Includes various fields such as alpha, q, beta, etc., which control the neuron tuning properties.

% OUTPUTS:
% - thetaMLE: [1 x numIntervals] The maximum likelihood estimate (MLE) of the stimulus orientation for each time window.
% ---------------------------------------------------------------------
function [thetaMLE, pdf, metrics] = decodePoissonSpikes(spikes, tuningFns, params)
    % Decode assuming poisson spiking
    
    uniqStimOris = params.uniqStimOris;
    stimDuration = params.stimDuration;     % Stimulus duration (total time)
    
    assert(sum( uniqStimOris >= -pi & uniqStimOris <= pi) == numel(uniqStimOris) );
    
    % Decode orientation from aggrgate spike count (no need to consider small individual time windows)
    aggregateSpikeCounts = sum(spikes, 2);
    
    % compute PDF
    n       = aggregateSpikeCounts;     % Spike count vector for all neurons in the current window
    f_theta = tuningFns;                % Firing rates for all neurons for each candidate orientation

    % Add a small epsilon to the firing rates to avoid numerical issues (log of 0)
    % TODO: make sure adding epsilon is okay
    temp     = stimDuration * f_theta;% + eps; 
    log_pdf  = sum(n .* log(temp), 1) - sum(temp, 1) - sum(gammaln(n+1), 1); 
    %log_pdf = sum(n .* log(temp), 1) - sum(temp, 1) - sum(log(factorial(n)), 1); 
    
    if any(~isfinite(log_pdf(:)))  
        % true if NaN or Inf
        keyboard
    end

    [~, idx]     = max(log_pdf(:));   % max value + linear index
    thetaMLE     = uniqStimOris(idx);
    
    pdf = exp(log_pdf - max(log_pdf(:)));
    normFactor = trapz(uniqStimOris, pdf);
    pdf = pdf / normFactor;
    
    % Uncertainty computation
    theta = uniqStimOris(:);   % in radians
    w     = pdf(:); % * dOri;      % weights
    C     = trapz(theta, w .* cos(2*theta));
    S     = trapz(theta, w .* sin(2*theta));
    
    mu    = 0.5 * atan2(S, C); mu = mod(mu, pi);
    R     = sqrt(C^2 + S^2); % V = 1 - R; % circular variance
    sigma = sqrt(-2 * log(R)) / 2; % circular standard deviation
    
    dOri   = uniqStimOris(2) - uniqStimOris(1); % in radians
    mu2    = sum( uniqStimOris(:).*pdf(:)*dOri );
    sigma2 = sqrt( sum( (uniqStimOris(:) - mu2).^2.*pdf(:)*dOri ) );
    
    metrics.mu     = mu;
    metrics.sigma  = sigma;
    metrics.mu2    = mu2;
    metrics.sigma2 = sigma2;
    
end