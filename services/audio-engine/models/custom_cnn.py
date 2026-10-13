import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    def __init__(self, input_dim, num_classes=2):
        self.model = CustomCNN(input_dim, num_classes)
        self.input_dim = input_dim

    def train(self, X, y):
        # Minimal training loop implementation
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            return torch.argmax(outputs, dim=1).numpy()
