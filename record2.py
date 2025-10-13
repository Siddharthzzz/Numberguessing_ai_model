import torch
import numpy as np
from torch import nn
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def p(*args):
    print(args)

p(torch.__version__)

### Setup device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Data ---
data = {
    "Size_SqFt": [
        1500, 2200, 1200, 1800, 2500, 1100, 1600, 3000, 1400, 
        2000, 2800, 1300, 1700, 2300, 1900, 2600, 1550, 2100
    ],
    "SalePrice_Lakhs": [
        95, 210, 70, 110, 250, 75, 150, 320, 85, 
        130, 280, 140, 98, 155, 180, 275, 92, 140
    ]
}

# --- Data Preparation ---
X = torch.tensor(data["Size_SqFt"], dtype=torch.float32).unsqueeze(1)
y = torch.tensor(data["SalePrice_Lakhs"], dtype=torch.float32).unsqueeze(1)

trainsplit = int(0.8 * len(X))
Xtrain, ytrain = X[:trainsplit], y[:trainsplit]
xtest, ytest = X[trainsplit:], y[trainsplit:]

scaler = StandardScaler()
X_train_scaled = torch.tensor(scaler.fit_transform(Xtrain), dtype=torch.float32)
X_test_scaled = torch.tensor(scaler.transform(xtest), dtype=torch.float32)

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

# --- Model ---
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ll(x)

torch.manual_seed(3)
model2 = Linear().to(device)
print(f"Model parameters before training: {model2.state_dict()}")
print(f"Model is on device: {next(model2.parameters()).device}")

# --- Before training ---
with torch.inference_mode():
    ypred_before = model2(X_test_scaled.to(device))
plotpredictions(Xtrain, ytrain, xtest, ytest, predictions=ypred_before.cpu())

# --- Training Setup ---
loss_fn = nn.L1Loss()
optimiser = torch.optim.SGD(params=model2.parameters(), lr=0.1)

# --- Training Loop ---
epochs = 7000
X_train_scaled_device = X_train_scaled.to(device)
ytrain_device = ytrain.to(device)
X_test_scaled_device = X_test_scaled.to(device)
ytest_device = ytest.to(device)

for epoch in range(epochs):
    model2.train()
    y_pred = model2(X_train_scaled_device)
    loss = loss_fn(y_pred, ytrain_device)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    model2.eval()
    with torch.inference_mode():
        testpred = model2(X_test_scaled_device)
        testloss = loss_fn(testpred, ytest_device)

    if epoch % 100 == 0:
        p(f"Epoch:{epoch} | Train loss: {loss:.4f} | Test loss: {testloss:.4f}")

print(f"Model parameters after training: {model2.state_dict()}")

# --- Final Predictions ---
model2.eval()
with torch.inference_mode():
    final_preds_gpu = model2(X_test_scaled_device)

final_preds_cpu = final_preds_gpu.cpu()

# Debug shapes
print("xtest shape:", xtest.shape)
print("ytest shape:", ytest.shape)
print("final preds shape:", final_preds_cpu.shape)

# Plot final predictions vs original sqft
plotpredictions(Xtrain, ytrain, xtest, ytest, predictions=final_preds_cpu)

# Print actual vs predicted
print("Actual test prices:", ytest.squeeze().numpy())
print("Predicted test prices:", final_preds_cpu.squeeze().numpy())

