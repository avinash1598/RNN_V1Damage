import os
import torch
import mat73
import numpy as np


class RNNDataLoader:
    """Simple data loader for RNN training.
    """
    def __init__(self, dataDIR: str = "/Users/gorislab/Desktop/Ranjan Workspace/RNN_V1Damage/V1DamageModel/"):
        self.dataDIR       = dataDIR
        self.trainFileName = "SpikeData.mat"
        self.testFileName  = "SpikeDataTest.mat"

        self.trainFilePath = os.path.join(self.dataDIR, self.trainFileName)
        self.testFilePath  = os.path.join(self.dataDIR, self.testFileName)

    def load_data(self):
        """
        Load test and train data from file path
        """
        self.data     = mat73.loadmat(self.trainFilePath, 'r')
        self.dataTest = mat73.loadmat(self.testFilePath, 'r')

    def get_train_data_intactV1(self):
        """
        Get training data as numpy arrays
        """
        spikeData      = self.data['data']['spikesIntactV1']
        tralStimVector = self.data['data']['trlStimVector']

        X, Y = self.format_data_for_RNN(tralStimVector, spikeData)
        
        return X, Y

    def get_test_data_intactV1(self):
        """
        Get test data as numpy arrays
        """
        # ── Load test data ───────────────────────────────────────────────────────────
        spikeData      = self.dataTest['dataTest']['spikesIntactV1Test']
        tralStimVector = self.data['data']['trlStimVector']

        X, Y = self.format_data_for_RNN(tralStimVector, spikeData)
        
        return X, Y

    def get_train_data_damagedV1(self):
        """
        Get training data as numpy arrays
        """
        spikeData      = self.data['data']['spikesDamagedV1']
        tralStimVector = self.data['data']['trlStimVector']

        X, Y = self.format_data_for_RNN(tralStimVector, spikeData)
        
        return X, Y
    
    def get_test_data_damagedV1(self):
        """
        Get test data as numpy arrays
        """
        # ── Load test data ───────────────────────────────────────────────────────────
        spikeData      = self.dataTest['dataTest']['spikesV1DamagedTest']
        tralStimVector = self.data['dataTest']['trlStimVector']

        X, Y = self.format_data_for_RNN(tralStimVector, spikeData)

        return X, Y
            
    def format_data_for_RNN(self, tralStimVector, spikeData):
        """
        Format data for RNN (convert to tensors)
        """
        tralStimVector = np.rad2deg(tralStimVector)
        correctLabels  = (tralStimVector > 90).astype(int)

        spikes = np.zeros((len(spikeData), len(spikeData[1]), len(spikeData[1][1])))
        for i in range(len(spikeData)):
            _spkData = spikeData[i]
            spikes[i, :, :] = _spkData.T

        spikes = spikes.astype(int) # ntrials x ntimePoints x nNeurons
        labels = correctLabels

        spikes = spikes.astype(np.float32)   # (4000, 200, 200)
        labels = labels.astype(np.float32).squeeze()  # (4000,)
        
        X = spikes
        Y = labels
        
        return X, Y
