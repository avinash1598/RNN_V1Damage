import torch.nn as nn
import torch.nn.functional as F

# ── model ─────────────────────────────────────────────────────────────────────
class DecisionRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn    = nn.RNN(input_size, hidden_size,
                              nonlinearity='tanh', batch_first=True)
        
        # readout heads
        self.w_dec  = nn.Linear(hidden_size, 1) # Decision readout
        self.w_conf = nn.Linear(hidden_size, 1) # Confidence readout
        
        # initialise weights
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

        # small uniform init for readout heads
        nn.init.uniform_(self.w_dec.weight,  -0.1, 0.1)
        nn.init.uniform_(self.w_conf.weight, -0.1, 0.1)
        
    def forward(self, x):
        """
        x : (batch, T, N_neurons)

        Returns
        -------
        z_dec  : (batch,)   decision logit
        z_conf : (batch,)   confidence logit
        h      : (batch, T, hidden)  full hidden trajectory (for analysis)
        """
        h, _   = self.rnn(x)
        h_T    = h[:, -1, :] # take the last time point's hidden state for readout
        z_dec  = self.w_dec(h_T).squeeze(-1) # Linear readout for decision
        z_conf = self.w_conf(h_T).squeeze(-1) # Linear readout for confidence
        return z_dec, z_conf, h
        # What is happening here - understand this