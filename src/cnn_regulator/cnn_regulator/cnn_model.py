import torch
import torch.nn as nn
import torch.nn.functional as F

class MIMO_CNN_Regulator(nn.Module):
    def __init__(self, sequence_length=10, num_sensors=12, num_actuators=6):
        super(MIMO_CNN_Regulator, self).__init__()
        # We assume input shape is (Batch, num_sensors, sequence_length)
        # Using 1D convolutions to extract temporal features from sensor time series
        
        self.conv1 = nn.Conv1d(in_channels=num_sensors, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Calculate the resulting flattened dimension
        self.flattened_dim = 32 * sequence_length 
        
        self.fc1 = nn.Linear(self.flattened_dim, 64)
        self.fc2 = nn.Linear(64, num_actuators)
        
    def forward(self, x):
        # x is (B, C, L) = (Batch, num_sensors, sequence_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        # Output layer for control actions (e.g., voltages, velocities, valve openings)
        out = self.fc2(x)
        
        return out
