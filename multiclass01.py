import torch
import numpy 
from sklearn.datasets import make_blobs
from torch import nn
import matplotlib.pyplot
matplotlib.use("TkAgg")   # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy
#### device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"uisng device: {device}")

#### importing a Dataset
### setting hte hyperparameters for the dataset 

numclasses = 4 
numfeatures = 2 
randomseeds = 42

# creatingn the multiclass data 
X_blob,y_blob = make_blobs(n_samples=1000,
                             n_features=numfeatures,
                             centers=numclasses,
                             cluster_std =1.5,
                             random_state=randomseeds)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_train,X_test,y_train,y_test = train_test_split(X_blob,y_blob,test_size=0.2,random_state= randomseeds)

###plotting 
plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0], X_blob[:,1],c=y_blob,cmap=plt.cm.RdYlBu)
plt.show()

#### buildig a multiclass classification model 

class blobmodel(nn.Module):
    def __init__(self,input_features,output_features,hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
         nn.Linear(in_features = input_features,out_features=hidden_units),
         nn.ReLU(),
         nn.Linear(in_features = hidden_units , out_features = hidden_units),
         nn.ReLU(),
         nn.Linear(in_features=hidden_units,out_features=output_features)
        )

    def forward(self,x):
        return self.linear_layer_stack(x)
    

model_mul = blobmodel(input_features=2,
                      output_features=4,
                      hidden_units=16).to(device)


loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(params=model_mul.parameters(),lr = 0.1)
metric = MulticlassAccuracy(num_classes=numclasses).to(device)

def accuracy(y_true,y_pred):
 correct = torch.eq(y_true,y_pred).sum().item()
 acc = (correct / len(y_pred))*100 
 return acc

### getting prediction probabilties for a multiclass pytorch model 
with torch.inference_mode():
   y_logits = model_mul(X_test.to(device))

y_pred_probs = torch.softmax(y_logits , dim=1)

y_preds = torch.argmax(y_pred_probs,dim = 1)

torch.manual_seed = randomseeds

epochs = 150
X_train,y_train = X_train.to(device),y_train.to(device)
X_test,y_test = X_test.to(device),y_test.to(device)
  
for epoch in range(epochs):
   model_mul.train()
   y_logits = model_mul(X_train)
   y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
   loss = loss_fn(y_logits,y_train)
   acc = accuracy(y_true=y_train,y_pred=y_pred)
   optimiser.zero_grad()
   loss.backward()
   optimiser.step()

   model_mul.eval()
   with torch.inference_mode():
      test_logits = model_mul(X_test)
      test_preds = torch.softmax(test_logits,dim=1).argmax(dim=1)
      test_loss = loss_fn(test_logits , y_test)
      test_acc = accuracy(y_true=y_test,y_pred=test_preds)


   if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Testloss: {test_loss:.4f} | testacc: {test_acc:.2f}%")


# ------------------------------
# 🟩 Classification Report
# ------------------------------
from sklearn.metrics import classification_report

model_mul.eval()
with torch.inference_mode():
    y_test_pred_logits = model_mul(X_test)
    y_test_preds = torch.softmax(y_test_pred_logits, dim=1).argmax(dim=1)

# move to CPU for sklearn
y_true = y_test.cpu().numpy()
y_pred = y_test_preds.cpu().numpy()

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, digits=4))

# torchmetrics accuracy (for confirmation)
torch_acc = metric(y_test_preds, y_test)
print(f"🔥 TorchMetrics Accuracy: {torch_acc*100:.2f}%")


def plot_decision_boundary(model, X, y):
    model.eval()
    with torch.inference_mode():
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 200),
                                torch.linspace(y_min, y_max, 200),
                                indexing="xy")
        grid = torch.cat([xx.reshape(-1,1), yy.reshape(-1,1)], dim=1).to(device)
        preds = model(grid)
        Z = torch.softmax(preds, dim=1).argmax(dim=1).reshape(xx.shape).cpu()

    plt.figure(figsize=(10,7))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor="k")
    plt.title("Decision Boundary (Multiclass)")
    plt.show()

# Call the plot function
plot_decision_boundary(model_mul, X_blob, y_blob)


print("actual " , y_test.squeeze().cpu().numpy())
print("pridicted " ,test_preds.squeeze().cpu().numpy())


### a few classification metrics 
#accuracy
#recall
#precision
#f1-score
#cofusion matirx
#classification report 
