import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
try:
    from .cnn_model import MIMO_CNN_Regulator
except ImportError:
    from cnn_model import MIMO_CNN_Regulator
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

class RobotDataset(Dataset):
    def __init__(
        self,
        csv_file,
        seq_len=10,
        num_sensors=12,
        num_actuators=6,
        state_scaler=None,
        action_scaler=None,
        fit_scalers=True,
    ):
        self.data = pd.read_csv(csv_file).values.astype(np.float32)
        self.seq_len = seq_len
        self.num_sensors = num_sensors
        self.num_actuators = num_actuators

        state_dim = self.seq_len * self.num_sensors
        self.states = self.data[:, :state_dim]
        self.actions = self.data[:, state_dim:state_dim + self.num_actuators]

        self.state_scaler = state_scaler if state_scaler is not None else StandardScaler()
        self.action_scaler = action_scaler if action_scaler is not None else StandardScaler()

        if fit_scalers:
            self.states = self.state_scaler.fit_transform(self.states)
            self.actions = self.action_scaler.fit_transform(self.actions)
        else:
            self.states = self.state_scaler.transform(self.states)
            self.actions = self.action_scaler.transform(self.actions)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        
        # Reshape to (num_sensors, sequence_length)
        states = np.reshape(states, (self.seq_len, self.num_sensors)).T
        
        return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            outputs = model(states)
            loss = criterion(outputs, actions)
            batch_size = states.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)

def train_model():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = RobotDataset('training_data.csv')
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    model = MIMO_CNN_Regulator().to(device)

    # SmoothL1 is more robust to occasional outliers in torque labels.
    criterion = nn.SmoothL1Loss(beta=0.5)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    epochs = 150
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    training_losses = []
    validation_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            optimizer.zero_grad()

            outputs = model(states)
            loss = criterion(outputs, actions)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = states.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / max(total_samples, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'state_scaler_mean': dataset.state_scaler.mean_.astype(np.float32),
                'state_scaler_scale': dataset.state_scaler.scale_.astype(np.float32),
                'action_scaler_mean': dataset.action_scaler.mean_.astype(np.float32),
                'action_scaler_scale': dataset.action_scaler.scale_.astype(np.float32),
                'sequence_length': dataset.seq_len,
                'num_sensors': dataset.num_sensors,
                'num_actuators': dataset.num_actuators,
                'best_val_loss': float(best_loss),
            }
            torch.save(checkpoint, 'cnn_regulator_weights_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                break

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'state_scaler_mean': dataset.state_scaler.mean_.astype(np.float32),
            'state_scaler_scale': dataset.state_scaler.scale_.astype(np.float32),
            'action_scaler_mean': dataset.action_scaler.mean_.astype(np.float32),
            'action_scaler_scale': dataset.action_scaler.scale_.astype(np.float32),
            'sequence_length': dataset.seq_len,
            'num_sensors': dataset.num_sensors,
            'num_actuators': dataset.num_actuators,
            'best_val_loss': float(best_loss),
        },
        'cnn_regulator_weights.pth'
    )
    print("Model saved to cnn_regulator_weights.pth")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(training_losses, linewidth=2, label='Training Loss')
    ax.plot(validation_losses, linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('CNN Regulator Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('training_loss_improved.png', dpi=150)
    print("Training loss visualization saved to training_loss_improved.png")

if __name__ == "__main__":
    train_model()
