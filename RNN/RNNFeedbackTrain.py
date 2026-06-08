import scipy.io as sio

# for .mat files saved in MATLAB v7.2 or earlier (most common)
data = sio.loadmat('/Users/avinashranjan/Desktop/UT Austin/Goris lab/Model_V1_damage/V1DamageModel/SpikeData.mat', simplify_cells=True)
# print(data['data'].keys())   # see what variables are inside

# Data structures for training
spikeData      = data['data']['spikesIntactV1']
tralStimVector = data['data']['trlStimVector']