import torch
import numpy as np
from torch import nn
import matplotlib.pyplot
matplotlib.use("TkAgg")   # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

dia = load_diabetes()
X,y = load_diabetes(return_X_y=True)

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=1)

scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)

xtrain = torch.tensor(Xtrain_scaled,dtype=torch.float).to(device)
xtest = torch.tensor(xtest_scaled,dtype=torch.float).to(device)
ytrain = torch.tensor(ytrain,dtype=torch.float).to(device)
ytest = torch.tensor(ytest,dtype=torch.float).to(device)

def plotpredictions(train_data = xtrain,
                    train_labels = ytrain,
                    test_data = xtest,
                    test_labels = ytest,
                    predictions = None
                    ):
    plt.figure(figsize = (10,7))
    # Use only the first feature (index 0) for plotting
    plt.scatter(train_data.cpu()[:, 0], train_labels.cpu(), c="b", s=4, label="training data")
    plt.scatter(test_data.cpu()[:, 0], test_labels.cpu(), c="g", s=4, label="testing data")
    if predictions is not None:
        plt.scatter(test_data.cpu()[:, 0], predictions.cpu(), c="r", s=4, label="Predictions")
    plt.xlabel("Feature 0")
    plt.ylabel("Target")
    plt.legend(prop = {"size" : 14})
    plt.show()


class dia(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=10,out_features=100)
        self.layer2 = nn.Linear(in_features=100,out_features=100)
        self.layer3 = nn.Linear(in_features=100,out_features=1)
        self.relu = nn.ReLU()  # Add activation function
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)  # Apply ReLU after first layer
        x = self.relu(self.layer2(x))
        x = self.dropout(x)  # Apply ReLU after second layer
        return self.layer3(x)  
modeldia = dia()

loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(modeldia.parameters(),
                            lr = 0.001,
                             weight_decay=1e-4) 


epochs = 5000
train_losses = []

for epoch in range(epochs):
    # Forward pass
    modeldia.train()
    y_pred = modeldia(xtrain).squeeze()
    loss = loss_fn(y_pred, ytrain.squeeze())

    # Backward pass
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    # Store and print progress
    train_losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Switch model to evaluation mode
modeldia.eval()

# Get predictions for the test set
with torch.no_grad():
    y_pred_test = modeldia(xtest).squeeze()

# Detach tensors from the GPU/computation graph and convert to numpy
xtest_numpy = xtest.cpu().numpy()
ytest_numpy = ytest.cpu().numpy()
y_pred_test_numpy = y_pred_test.cpu().numpy()

# Get the first feature from the test data
feature_index = 0
feature_to_plot = xtest_numpy[:, feature_index]

# --- Plotting ---
plt.figure(figsize=(10, 7))

# 1. Plot the actual data points (ground truth)
plt.scatter(feature_to_plot, ytest_numpy, c="blue", label="Actual Values")

# 2. Plot the model's predictions
plt.scatter(feature_to_plot, y_pred_test_numpy, c="red", label="Predictions")

plt.xlabel(f"Feature {feature_index}")
plt.ylabel("Disease Progression")
plt.title("Model Predictions vs. Actual Data")
plt.legend()
plt.show()

print("actual " , ytest.squeeze().numpy())
print("pridicted " , y_pred_test.squeeze().numpy())