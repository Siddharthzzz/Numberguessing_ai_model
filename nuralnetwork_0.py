import torch
import numpy 
from sklearn.datasets import make_circles
from torch import nn
import matplotlib.pyplot
matplotlib.use("TkAgg")   # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"uisng device: {device}")

nsamples = 10000

X,y = make_circles(nsamples,noise=0.03,random_state=42) 
print(len(X),len(y))
print(f"First 5 samples of x:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

circles = pd.DataFrame({"X1" : X[:,0],
                        "X2" : X[:,1],
                        "label" : y})

print(circles.head(10))
plt.scatter(x=X[:,0],
            y=X[:,1],
             c=y,
             cmap = plt.cm.RdYlBu)


X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

from sklearn.model_selection import train_test_split

Xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,
                                        random_state=42)

class nural1(nn.Module):
    def __init__(self):
     super().__init__()
     self.layer_1 = nn.Linear(in_features=2,out_features=16) ## input layer 
     self.relu1 = nn.ReLU()
     self.layer2 = nn.Linear(in_features=16,out_features=8) 
     self.relu2 = nn.ReLU()
     self.layer3 = nn.Linear(in_features=8,out_features=1) ## output layer 
    def forward(self,x):
        x = self.relu1(self.layer_1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)  # raw logits
        return x
    

class nural2(nn.Module):
  def __init__(self):
   super().__init__()
   self.input_layer = nn.Linear(in_features=2,out_features=10) 
   self.layer1 = nn.Linear(in_features=10,out_features=10)
   self.output_layer = nn.Linear(in_features=10 , out_features=1)
  def forward(self,x):
    z = self.input_layer(x)
    z = self.layer1(z)
    z = self.output_layer(z)
    return z


model_nural = nural1().to(device)

###using nn.seqential
#model_nural = nn.Sequential(
   #nn.Linear(in_features=2,out_features=8),
   #nn.Linear(in_features=8,out_features=1)
   
#).to(device)
#print(model_nural)

#print(model_nural.state_dict())  
with torch.inference_mode():
 untrained_preds = model_nural(xtest.to(device))
print(f"Length of predictions: {len(untrained_preds)} , shape: {untrained_preds.shape}")
print(f"length of test samples:{len(xtest)},shape: {xtest.shape}")
print(f"\nfirst 10 preds:\n{torch.round(untrained_preds[:10])}")

### setting up loss and optimiser 

loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(params=model_nural.parameters(),
                            lr=0.01)

##calucaltin the accuracy 
def accuracy_fn(y_true,y_pred):
  correct = torch.eq(y_true,y_pred).sum().item()
  acc = (correct/len(y_pred)) * 100
  return acc

### traning a model 
## going from raw logits to pred prob to pred labels 
model_nural.eval()
with torch.inference_mode():
  y_logits = model_nural(xtest.to(device))[:5]
print(y_logits)

##using sigmod function for activation 

y_pred_probs = torch.sigmoid(y_logits)

y_pred = torch.round(y_pred_probs)
## same thing as the above but in full 
y_labels = torch.round(torch.sigmoid(model_nural(xtest.to(device))[:5]))

## check for equality
print(torch.eq(y_pred.squeeze(),y_labels.squeeze()))
## geting rid of extra dimention 
y_pred.squeeze()

torch.cuda.manual_seed(1)

###traning loop 
epochs = 1000
Xtrain,ytrain = Xtrain.to(device),ytrain.to(device)
xtest,ytest = xtest.to(device),ytest.to(device)
  
for epoch in range(epochs):
  model_nural.train()
  ###forward pass
  y_logits = model_nural(Xtrain).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))

  ##calucalting the loss and the accurecy 
  ##loss = loss_fn(torch.sigmoid(y_logits),ytrain)
  loss = loss_fn(y_logits,ytrain)
  acc = accuracy_fn(y_true=ytrain,y_pred=y_pred)
  ##optimiser 
  optimiser.zero_grad()
  loss.backward()
  optimiser.step()

 ###testing

  model_nural.eval()
  with torch.inference_mode():
    test_logits = model_nural(xtest).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))

    test_loss = loss_fn(test_logits,ytest)
    test_acc = accuracy_fn(y_true=ytest,y_pred=test_pred)

 ###printing whats happening 
  if epoch % 10 == 0:
   print(f"Epoch: {epoch} | Loss: {loss:.4f} | ACC:{acc:.2f}% | Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")


# Plot decision boundary function
import numpy as np

def plot_decision_boundary(model, X, y):
    plt.close("all")            # Close any previous figures
    plt.figure(figsize=(6,6))   # New figure
    
    model.eval()
    with torch.inference_mode():
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
        preds = torch.sigmoid(model(grid))
        preds = torch.round(preds).cpu().numpy()
        Z = preds.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor="k")
    plt.title("Decision Boundary")
    plt.show()


# Call after training
plot_decision_boundary(model_nural, X.cpu(), y.cpu())


print("actual " , ytest.squeeze().cpu().numpy())
print("pridicted " ,test_pred.squeeze().cpu().numpy())

# Assuming you have a trained model_nural
model_nural.eval()
a = float(input("Enter a: "))
b = float(input("Enter b: "))
# Create a new data point as a tensor
new_data = torch.tensor([[a, b]]).to(device) # Example input

# Get the prediction
with torch.inference_mode():
    prediction = model_nural(new_data)

# Print the prediction (raw output, or "logits")
print(f"The model's raw prediction is: {prediction.item():.4f}")

# Convert the raw prediction to a probability and then a class label
prediction_prob = torch.sigmoid(prediction)
prediction_label = torch.round(prediction_prob)

print(f"The predicted probability is: {prediction_prob.item():.4f}")
print(f"The predicted class label is: {prediction_label.item()}")

