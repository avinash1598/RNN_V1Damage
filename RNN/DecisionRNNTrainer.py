import torch
import torch.nn as nn
import torch.nn.functional as F

class DecisionRNNTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_epoch(self, n_trials, X, Y, n_epochs: int = 20, batch_size: int = 64):
        X = torch.tensor(X)   # (4000, 200, 200)
        Y = torch.tensor(Y)   # (4000,)

        self.model.train() # set model to training mode

        for epoch in range(n_epochs):
    
            # shuffle each epoch
            perm     = torch.randperm(n_trials)
            X_, Y_   = X[perm], Y[perm]
            
            epoch_loss = 0
            n_correct  = 0

            for i in range(0, n_trials, batch_size):
                x_batch = X_[i : i+batch_size]
                y_batch = Y_[i : i+batch_size]
                
                self.optimizer.zero_grad()
                z_dec, z_conf, h = self.model(x_batch)

                # choice loss
                p_choice = torch.where(z_dec > 0, torch.sigmoid(z_dec), 1 - torch.sigmoid(z_dec))
                L_choice = F.binary_cross_entropy_with_logits(z_dec, y_batch)

                # brier loss
                with torch.no_grad():
                    correct = ((z_dec > 0).float() == y_batch).float()
                    lossWeights = torch.where(correct == 1, 20, 20) #18.8, 40
                c_hat    = torch.sigmoid(z_conf)
                L_conf   = torch.mean(lossWeights * ( (correct - c_hat)**2 )) #F.mse_loss(c_hat, correct)

                loss = L_choice + L_conf # 20, 100*L_conf + L_z_reg % change score - lc and hc should probably be treated differently
                loss.backward() # compute gradient
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # normalize gradients to prevent exploding gradients
                self.optimizer.step() # Update weights
                
                epoch_loss += loss.item()
                n_correct  += ((z_dec > 0).float() == y_batch).sum().item()

                acc = n_correct / n_trials
                print(f"Epoch {epoch+1:3d} | loss: {epoch_loss:.4f} | acc: {acc:.3f}")

    