clear all
close all
clc

rng('shuffle');

addpath('/Users/avinashranjan/Desktop/UT Austin/Goris lab/Model_V1_damage/V1DamageModel/Scripts/')

load('SpikeData.mat')

trialDecisions          = data.trialDecisions;
trialConfs              = data.trialConfs;
trlStimVector           = data.trlStimVector;  % Stim vector for each trial
spikesIntactV1          = data.spikesIntactV1;
