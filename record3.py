import torch
import numpy as np
from torch import nn
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# fetch dataset (Auto MPG is dataset ID = 9)
auto_mpg = fetch_ucirepo(id=9)

# Features (inputs)
X = auto_mpg.data.features

# Target (output: mpg)
y = auto_mpg.data.targets  

# Metadata (info about dataset)
print(auto_mpg.metadata)

# Variables (column names + details)
print(auto_mpg.variables)

### Setup device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Fill missing values in horsepower with median
X["horsepower"] = X["horsepower"].fillna(X["horsepower"].median())

# Train/test split BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features using only training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
y_test  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

# --- Plotting Function ---
def plotpredictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data.cpu().squeeze(), train_labels.cpu().squeeze(),
                c="b", s=20, label="training data")
    plt.scatter(test_data.cpu().squeeze(), test_labels.cpu().squeeze(),
                c="g", s=20, label="testing data")
    if predictions is not None:
        # Plot predictions against original test_data
        plt.scatter(test_data.cpu().squeeze(), predictions.cpu().squeeze(),
                    c="r", s=20, label="Predictions")
    plt.xlabel("Size (sqft)")
    plt.ylabel("Sale Price (Lakhs)")
    plt.legend(prop={"size": 14})
    plt.show()

class MPGModel(nn.Module):
    def __init__(self):
        super(MPGModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(7, 64),   # input → hidden1
            nn.ReLU(),
            nn.Linear(64, 32),  # hidden1 → hidden2
            nn.ReLU(),
            nn.Linear(32, 1)    # hidden2 → output
        )

    def forward(self, x):
        return self.layers(x)

# Create model instance
model = MPGModel().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5000
train_losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store and print progress
    train_losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    test_loss = loss_fn(y_pred_test, y_test).item()

print(f"\nFinal Test Loss (MSE): {test_loss:.4f}")

# Plotting the predicted vs. actual values with requested colors
plt.figure(figsize=(10, 7))

# Predictions for the training data
plt.scatter(y_train.cpu().numpy(), y_train.cpu().numpy(), alpha=0.7, c="blue", label="Training Predictions")
# Predictions for the testing data
plt.scatter(y_test.cpu().numpy(), y_pred_test.cpu().numpy(), alpha=0.7, c="red", label="Test Predictions")

plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Predicted vs Actual MPG")
plt.legend()
plt.show()

print("Actual test prices:", y_test.squeeze().numpy())
print("Predicted test prices:", y_pred_test.squeeze().numpy())