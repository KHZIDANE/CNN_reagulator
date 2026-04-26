import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from cnn_model import MIMO_CNN_Regulator
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class RobotDataset(Dataset):
    def __init__(self, csv_file, seq_len=10, num_sensors=12, scaler=None):
        self.data = pd.read_csv(csv_file).values
        self.seq_len = seq_len
        self.num_sensors = num_sensors
        self.scaler = scaler
        
        # Normalize data for better training
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.data = self.scaler.transform(self.data)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        # First 40 elements are state history (seq_len * num_sensors)
        states = row[:self.seq_len * self.num_sensors]
        actions = row[self.seq_len * self.num_sensors:]
        
        # Reshape to (num_sensors, sequence_length)
        states = np.reshape(states, (self.seq_len, self.num_sensors)).T
        
        return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)

def train_model():
    # Create dataset and dataloader with data normalization
    dataset = RobotDataset('training_data.csv')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)  # Increased batch size
    
    model = MIMO_CNN_Regulator()
    
    # Use a combination of losses for better regularization
    criterion = nn.MSELoss()
    
    # Use AdamW for better weight decay and optimization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # More aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
    
    epochs = 100  # Increased epochs with early stopping
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    training_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for states, actions in dataloader:
            optimizer.zero_grad()
            
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count
        training_losses.append(avg_loss)
        
        # Exponential moving average for smoother logging
        if epoch > 0:
            ema_loss = 0.9 * training_losses[-2] + 0.1 * avg_loss
        else:
            ema_loss = avg_loss
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, EMA Loss: {ema_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'cnn_regulator_weights_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                break
        
    torch.save(model.state_dict(), 'cnn_regulator_weights.pth')
    print("Model saved to cnn_regulator_weights.pth")
    
    # Create training loss visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(training_losses, linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Improved CNN Regulator Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('training_loss_improved.png', dpi=150)
    print("Training loss visualization saved to training_loss_improved.png")

if __name__ == "__main__":
    train_model()
