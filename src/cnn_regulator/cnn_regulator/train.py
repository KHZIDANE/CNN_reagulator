import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from cnn_model import MIMO_CNN_Regulator
import numpy as np

class RobotDataset(Dataset):
    def __init__(self, csv_file, seq_len=10, num_sensors=12):
        self.data = pd.read_csv(csv_file).values
        self.seq_len = seq_len
        self.num_sensors = num_sensors
        
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
    dataset = RobotDataset('training_data.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MIMO_CNN_Regulator()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for states, actions in dataloader:
            optimizer.zero_grad()
            
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), 'cnn_regulator_weights.pth')
    print("Model saved to cnn_regulator_weights.pth")

if __name__ == "__main__":
    train_model()
