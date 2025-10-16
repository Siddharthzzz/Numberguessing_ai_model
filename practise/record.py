import torch
import numpy as np
from torch import nn
import matplotlib.pyplot
matplotlib.use("TkAgg")   # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def p(*args):
    print(args)



p(torch.__version__)

###Setup device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"uisng device: {device}")


###Creating some data usisng linear regression 

weight = 0.8
bias = 0.2

#cretaing range vlues 
start = 0
end = 1
step = 0.02
 
#creating X and y (featuers and labels)

X = torch.arange(start , end , step).unsqueeze(dim=1)
y = weight*X + bias

###splitting data
train_pslit = int(0.8*len(X))
Xtrain = X[:train_pslit]
ytrain = y[:train_pslit]
xtest,ytest = X[train_pslit:] , y[train_pslit:]

#ploting the data 
def plotpredictions(train_data = Xtrain,
                    train_labels = ytrain,
                    test_data = xtest,
                    test_labels = ytest,
                    predictions = None
                    ):
    plt.figure(figsize = (10,7))
    plt.scatter(train_data.cpu(),train_labels.cpu(),c="b",s=4,label="training data")
    plt.scatter(test_data.cpu(),test_labels.cpu(),c = "g", s = 4,label="testing data")
    if predictions is not None:
        plt.scatter(test_data.cpu(),predictions.cpu() , c="r",s=4,label="Predictions")
    plt.legend(prop = {"size" : 14})
    plt.show()
##plotpredictions(Xtrain,ytrain,xtest,ytest)

###Creating a model 

class linear2(nn.Module):
    def __init__(self):
        super().__init__()
        ## using nn.linear() for creating the medel parameters
        self.linear_layer = nn.Linear(in_features=1,out_features=1)
        
    def forward(self,x:torch.tensor) ->torch.tensor:
        return self.linear_layer(x)

torch.manual_seed(1)
model_1 = linear2()

with torch.inference_mode(): #doesnt use grad which is used fo rtrsning cuz it will speed up the output
    ypred = model_1(xtest)

print(ypred)
plotpredictions(predictions=ypred)

model_1.to(device)
p(model_1.state_dict())
p(next(model_1.parameters()).device)


### Traning code

loss_fn = nn.L1Loss()
optimiser = torch.optim.SGD(params = model_1.parameters(),
                            lr=0.01)


torch.manual_seed(1)

epochs = 200

epoch_count = []
loss1_values = []
test_loss_values = []

Xtrain= Xtrain.to(device)
ytrain = ytrain.to(device)
xtest = xtest.to(device)
ytest = ytest.to(device)

for epoch in range(epochs):
    model_1.train()
    y_pred = model_1(Xtrain)
    loss2 = loss_fn(y_pred , ytrain)
    optimiser.zero_grad()
    loss2.backward()
    optimiser.step()
    ###testing 
    model_1.eval()
    with torch.inference_mode():
        testpred = model_1(xtest.to(device))
        testloss  = loss_fn(testpred , ytest)
        testpred = testpred.cpu()
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss1_values.append(loss2.detach().cpu().numpy())
        test_loss_values.append(testloss.detach().cpu().numpy())
        p(f"Epoch:{epoch} | loss: {loss2} | testloss: {testloss}")

p(model_1.state_dict())

model_1.eval()
with torch.inference_mode():
    # Move test data to GPU for final prediction
    final_preds_gpu = model_1(xtest.to(device))

# Move predictions BACK to the CPU for plotting
final_preds_cpu = final_preds_gpu.cpu()
plotpredictions(Xtrain, ytrain, xtest, ytest, predictions=final_preds_cpu)

# Plot loss curves
plt.plot(epoch_count, loss1_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and Test Loss Curve")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

