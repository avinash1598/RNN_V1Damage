clear all
close all
restoredefaultpath
clc

rng('shuffle');

addpath('/Users/avinashranjan/Desktop/UT Austin/Goris lab/Model_V1_damage/V1DamageModel/Scripts/')
addpath('/Users/gorislab/Desktop/Ranjan Workspace/RNN_V1Damage/V1DamageModel/Scripts/')

% ----------------------------------
% Params
% ----------------------------------
nNeurons     = 200;   % Count of neurons
stimDuration = 2;     % Stimulus duration set to 1 seconds
timeStep     = 0.001; % 0.001s (1ms) - Step size of time bins used for binning stimulus duration 

% Stimulus parameters (angles in radians)
contrasts               = [0.01]; %[1e-4 0.001 0.01 0.05 0.1 0.2 0.5]; % 0.01
spreads                 = [5]; %[3];
uniqStimOris            = linspace(90-11, 90+11, 21); %88:0.2:92; 88:0.2:92;
uniqStimOris            = deg2rad(uniqStimOris');
stimParam.numStim       = numel(uniqStimOris);                                % Number of unique stimuli
stimParam.countPerStim  = 200; % 100 
ntrials                 = stimParam.numStim * stimParam.countPerStim * numel(contrasts) * numel(spreads);  % Total number of trials

load('SpikeData.mat')
load('SpikeDataTest.mat')

trialDecisions          = data.trialDecisions;
trialDecisionsDamaged   = data.trialDecisionsDamaged;
trialDecisionsAdjusted  = data.trialDecisionsAdjusted;
trialConfs              = data.trialConfs;
trialConfsDamaged       = data.trialConfsDamaged;
trialConfsAdjusted      = data.trialConfsAdjusted;
trialPDFs               = data.trialPDFs;
trialPDFsDamaged        = data.trialPDFsDamaged;
trialPDFsAdjusted       = data.trialPDFsAdjusted;
trlContrastVector       = data.trlContrastVector;
trlSpreadVector         = data.trlSpreadVector;
trlStimVector           = data.trlStimVector;  % Vector of stimuli
confVars                = data.confVars;

% % Compute trial confidence
% x = confVars(trlStimVector == pi/2); Cc = median(x);
% trialConfs              = confVars > Cc;

% trialDecisions          = dataTest.trialDecisionsTest;
% trialConfs              = dataTest.trialConfsTest;

%% Psychometric function
% Intact
[psycFns, confFn, choiceCounts] = getPsychometricData( ...
    trialDecisions, ...
    trialConfs, ...
    stimParam, ...
    uniqStimOris, ...
    trlStimVector);

figure
subplot(3, 5, 1)
plot(rad2deg(uniqStimOris), psycFns(1, :))
xline(90, LineWidth=1.5, LineStyle="--")
ylim([0 1])
xlabel('Orientation')
ylabel('P(CW)')
title('Intact V1')

subplot(3, 5, 2)
scatter(rad2deg(uniqStimOris), psycFns(2, :), 200*(choiceCounts(1, :)+eps), 'filled', 'DisplayName', 'HC', MarkerFaceColor='blue');
hold on
scatter(rad2deg(uniqStimOris), psycFns(3, :), 200*(choiceCounts(2, :)+eps), 'filled', 'DisplayName', 'LC', MarkerFaceColor='red')
xline(90, LineWidth=1.5, LineStyle="--", HandleVisibility='off')
plot(rad2deg(uniqStimOris), psycFns(2, :), Color='blue', HandleVisibility='off');
plot(rad2deg(uniqStimOris), psycFns(3, :), Color='red', HandleVisibility='off');
hold off
ylim([0 1])
xlabel('Orientation')
ylabel('P(CW)')
legend

subplot(3, 5, 5)
scatter(rad2deg(uniqStimOris), psycFns(4, :), 200*(choiceCounts(3, :)+eps), 'filled', 'DisplayName', 'HC', MarkerFaceColor='blue');
hold on
scatter(rad2deg(uniqStimOris), psycFns(5, :), 200*(choiceCounts(4, :)+eps), 'filled', 'DisplayName', 'LC', MarkerFaceColor='red')
xline(90, LineWidth=1.5, LineStyle="--", HandleVisibility='off')
plot(rad2deg(uniqStimOris), psycFns(4, :), Color='blue', HandleVisibility='off');
plot(rad2deg(uniqStimOris), psycFns(5, :), Color='red', HandleVisibility='off');
hold off
ylim([0 1])
xlabel('Orientation')
ylabel('P(CCW)')
legend

subplot(3, 5, 3)
plot(rad2deg(uniqStimOris), confFn)
ylim([0 1])
xlabel('Orientation')
ylabel('P(HC)')

subplot(3, 5, 4)
scatter(psycFns(1, :), confFn)
ylim([0 1])
xlabel('P(CW)')
ylabel('P(HC)')

% Damaged
[psycFns, confFn, choiceCounts] = getPsychometricData( ...
    trialDecisionsDamaged, ...
    trialConfsDamaged, ...
    stimParam, ...
    uniqStimOris, ...
    trlStimVector);

subplot(3, 5, 6)
plot(rad2deg(uniqStimOris), psycFns(1, :))
xline(90, LineWidth=1.5, LineStyle="--")
ylim([0 1])
xlabel('Orientation')
ylabel('P(CW)')
title('Damaged V1')

subplot(3, 5, 7)
scatter(rad2deg(uniqStimOris), psycFns(2, :), 200*(choiceCounts(1, :)+eps), 'filled', 'DisplayName', 'HC', MarkerFaceColor='blue');
hold on
scatter(rad2deg(uniqStimOris), psycFns(3, :), 200*(choiceCounts(2, :)+eps), 'filled', 'DisplayName', 'LC', MarkerFaceColor='red')
xline(90, LineWidth=1.5, LineStyle="--", HandleVisibility='off')
plot(rad2deg(uniqStimOris), psycFns(2, :), Color='blue', HandleVisibility='off');
plot(rad2deg(uniqStimOris), psycFns(3, :), Color='red', HandleVisibility='off');
hold off
ylim([0 1])
xlabel('Orientation')
ylabel('P(CW)')
legend

subplot(3, 5, 8)
plot(rad2deg(uniqStimOris), confFn)
ylim([0 1])
xlabel('Orientation')
ylabel('P(HC)')

subplot(3, 5, 9)
scatter(psycFns(1, :), confFn)
ylim([0 1])
xlabel('P(CW)')
ylabel('P(HC)')

subplot(3, 5, 10)
scatter(rad2deg(uniqStimOris), psycFns(4, :), 200*(choiceCounts(3, :)+eps), 'filled', 'DisplayName', 'HC', MarkerFaceColor='blue');
hold on
scatter(rad2deg(uniqStimOris), psycFns(5, :), 200*(choiceCounts(4, :)+eps), 'filled', 'DisplayName', 'LC', MarkerFaceColor='red')
xline(90, LineWidth=1.5, LineStyle="--", HandleVisibility='off')
plot(rad2deg(uniqStimOris), psycFns(4, :), Color='blue', HandleVisibility='off');
plot(rad2deg(uniqStimOris), psycFns(5, :), Color='red', HandleVisibility='off');
hold off
ylim([0 1])
xlabel('Orientation')
ylabel('P(CCW)')
legend

% Adjusted
[psycFns, confFn, choiceCounts] = getPsychometricData( ...
    trialDecisionsAdjusted, ...
    trialConfsAdjusted, ...
    stimParam, ...
    uniqStimOris, ...
    trlStimVector);

subplot(3, 5, 11)
plot(rad2deg(uniqStimOris), psycFns(1, :))
xline(90, LineWidth=1.5, LineStyle="--")
ylim([0 1])
xlabel('Orientation')
ylabel('P(CW)')
title('Adjusted V1')

subplot(3, 5, 12)
scatter(rad2deg(uniqStimOris), psycFns(2, :), 200*(choiceCounts(1, :)+eps), 'filled', 'DisplayName', 'HC', MarkerFaceColor='blue');
hold on
scatter(rad2deg(uniqStimOris), psycFns(3, :), 200*(choiceCounts(2, :)+eps), 'filled', 'DisplayName', 'LC', MarkerFaceColor='red')
xline(90, LineWidth=1.5, LineStyle="--", HandleVisibility='off')
plot(rad2deg(uniqStimOris), psycFns(2, :), Color='blue', HandleVisibility='off');
plot(rad2deg(uniqStimOris), psycFns(3, :), Color='red', HandleVisibility='off');
hold off
ylim([0 1])
xlabel('Orientation')
ylabel('P(CW)')
legend

subplot(3, 5, 13)
plot(rad2deg(uniqStimOris), confFn)
ylim([0 1])
xlabel('Orientation')
ylabel('P(HC)')

subplot(3, 5, 14)
scatter(psycFns(1, :), confFn)
ylim([0 1])
xlabel('P(CW)')
ylabel('P(HC)')

subplot(3, 5, 15)
scatter(rad2deg(uniqStimOris), psycFns(4, :), 200*(choiceCounts(3, :)+eps), 'filled', 'DisplayName', 'HC', MarkerFaceColor='blue');
hold on
scatter(rad2deg(uniqStimOris), psycFns(5, :), 200*(choiceCounts(4, :)+eps), 'filled', 'DisplayName', 'LC', MarkerFaceColor='red')
xline(90, LineWidth=1.5, LineStyle="--", HandleVisibility='off')
plot(rad2deg(uniqStimOris), psycFns(4, :), Color='blue', HandleVisibility='off');
plot(rad2deg(uniqStimOris), psycFns(5, :), Color='red', HandleVisibility='off');
hold off
ylim([0 1])
xlabel('Orientation')
ylabel('P(CCW)')
legend

%%
function [psycFns, confFn, choiceCounts] = getPsychometricData( ...
    trialDecisions, ...
    trlConfReports, ...
    stimParam, ...
    uniqStimOris, ...
    trlStimVector)

    confFn       = zeros(1, stimParam.numStim);
    psycFns      = zeros(5, stimParam.numStim); % Combined, HC, LC
    choiceCounts = zeros(4, stimParam.numStim); % only for HC and LC
    
    for i=1:stimParam.numStim
        stimOrientation = uniqStimOris(i);
        givenOrientationTrialIDxes = find(trlStimVector == stimOrientation);
    
        decisions   = trialDecisions(givenOrientationTrialIDxes);
        confReports = trlConfReports(givenOrientationTrialIDxes);
    
        % Intact V1 population
        percent_CW    = length(find(decisions == 1)) / numel(decisions);
        psycFns(1, i) = percent_CW; 
        confFn(1, i)  = length(find(confReports == 1)) / numel(confReports);
        
        percent_CW_HC = length(find(decisions == 1 & confReports == 1)) / sum(confReports == 1);
        percent_CW_LC = length(find(decisions == 1 & confReports == 0)) / sum(confReports == 0);
        psycFns(2, i) = percent_CW_HC;
        psycFns(3, i) = percent_CW_LC;
        
        percent_CCW_HC = length(find(decisions == -1 & confReports == 1)) / sum(confReports == 1);
        percent_CCW_LC = length(find(decisions == -1 & confReports == 0)) / sum(confReports == 0);
        psycFns(4, i) = percent_CCW_HC;
        psycFns(5, i) = percent_CCW_LC;
        
        % choice count
        choiceCounts(1, i) = sum( ( (decisions == 1) & (confReports == 1) ) ) / numel(decisions) ;  %(CW, HC)
        choiceCounts(2, i) = sum( ( (decisions == 1) & (confReports == 0) ) ) / numel(decisions);  %(CW, LC)
        choiceCounts(3, i) = sum( ( (decisions == -1) & (confReports == 1) ) ) / numel((decisions)); %(CCW, HC)
        choiceCounts(4, i) = sum( ( (decisions == -1) & (confReports == 0) ) ) / numel((decisions)); %(CCW, LC)

    end

end
