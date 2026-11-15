import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class CustomCNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * (input_dim // 2), 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dim
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AudioCustomCNN:
    def __init__(self, input_dim, num_classes=2, learning_rate=0.001, epochs=30, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomCNN(input_dim, num_classes).to(self.device)
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X, y):
        """Fit with weighted loss and mini-batches on GPU when available."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        class_counts = np.bincount(y, minlength=2)
        weights = len(y) / (len(class_counts) * class_counts)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        train_loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        self.model.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch.to(self.device))
                loss = criterion(outputs, y_batch.to(self.device))
                loss.backward()
                optimizer.step()

    def predict(self, X, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)

            # Convert raw scores to probabilities
            probs = torch.softmax(outputs, dim=1)

            # Use the threshold on the 'abnormal' class (index 1)
            return (probs[:, 1] >= threshold).int().numpy()

    def save(self, path):
        """Saves the model weights to a file."""
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load(self, path):
        """Loads the model weights from a file."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model weights loaded from {path}")
