import torch
import torch.nn as nn
import torch.nn.functional as F

class MIMO_CNN_Regulator(nn.Module):
    def __init__(self, sequence_length=10, num_sensors=12, num_actuators=6):
        super(MIMO_CNN_Regulator, self).__init__()
        # We assume input shape is (Batch, num_sensors, sequence_length)
        # Using 1D convolutions to extract temporal features from sensor time series
        
        # Enhanced convolutional layers with residual-like connections
        self.conv1 = nn.Conv1d(in_channels=num_sensors, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Global average pooling reduces overfitting
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate the resulting flattened dimension (128 after pooling)
        self.flattened_dim = 128
        
        # Enhanced fully connected layers with better regularization
        self.fc1 = nn.Linear(self.flattened_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_actuators)
        
    def forward(self, x):
        # x is (B, C, L) = (Batch, num_sensors, sequence_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling instead of flattening
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Enhanced FC layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output layer for control actions (e.g., voltages, velocities, valve openings)
        out = self.fc3(x)
        
        return out
