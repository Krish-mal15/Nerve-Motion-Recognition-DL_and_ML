import numpy as np
import pandas as pd  # For loading and processing the dataset
from sklearn.model_selection import train_test_split
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

data = pd.read_csv('EMG-data.csv')
data = data.drop(['label', 'time'], axis=1)

X_train = data.drop('class', axis=1).values
y_train = data['class'].values

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))
print(X_train.shape, y_train.shape)
print(data.head())


class EMGffModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        torch.manual_seed(0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 8),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        return self.net(X)

    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred


input_dim = 8
model = EMGffModel(8)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)


# Training loop
def train_model(model, X_train, y_train, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


# Train the model
train_model(model, X_train.float(), y_train.long(), optimizer, loss_fn)

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        # Get predictions
        y_pred = model(X_test)
        # Convert predicted probabilities to predicted class labels
        _, predicted = torch.max(y_pred, 1)
        # Calculate accuracy
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total
        print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Evaluate the model
evaluate_model(model, X_test.float(), y_test.long())
