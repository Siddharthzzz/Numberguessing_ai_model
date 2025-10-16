import torch
import numpy as np
from torch import nn
import matplotlib
import torch.optim.sgd
matplotlib.use("TkAgg")   # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
from pathlib import Path

def p(*args):
    print(args)


p(torch.__version__)

## creating a dataset using linear regressions \\

#creating known parameters
weight = 0.7
bias = 0.3

#create data
### linear regression formula Y = f(X,b) + e
start = 0
end = 1
step = 0.02
X = torch.arange(start,end ,step).unsqueeze(dim=1)
y = weight * X + bias
p(X[:10] , y[:10] , len(X),len(y))



## splitting the datta in to training and test sets

trainsplit = int(0.8 * len(X))
Xtrain,ytrain = X[:trainsplit],y[:trainsplit]
xtest,ytest = X[trainsplit:],y[trainsplit:]
p(len(Xtrain) , len(ytrain),len(xtest),len(ytest))

def plotpredictions(train_data = Xtrain,
                    train_labels = ytrain,
                    test_data = xtest,
                    test_labels = ytest,
                    predictions = None
                    ):
    plt.figure(figsize = (10,7))
    plt.scatter(train_data,train_labels,c="b",s=4,label="training data")
    plt.scatter(test_data,test_labels,c = "g", s = 4,label="testing data")
    if predictions is not None:
        plt.scatter(test_data,predictions , c="r",s=4,label="Predictions")
    plt.legend(prop = {"size" : 14})
    plt.show()

    
## creating a linear regressions model class
''' whats our model does 
start with randone values (weights & bias)
look at ttraining data and adjust the random values to better
represent (or get closer to ) the ideal values (the weight &biaas 
to vreate data 
)

Two main algorithms:
1:Gradient descent
2:Back propogation
'''
class linear(nn.Module):
    def __init__(self):
        super().__init__() #call the parent constructor
        self.weight = nn.Parameter(torch.rand(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias =  nn.Parameter(torch.rand(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        ##forward method to define the coompputation in the model
    def forward(self,x:torch.tensor) -> torch.tensor:
        return self.weight *x + self.bias
        
torch.manual_seed(42)
model_0 = linear()
p(model_0)
p(list(model_0.parameters())) #To see the list and tensors 
p(model_0.state_dict()) #To see the names 

### Making presictions usiing "torch.inference_mode()"
with torch.inference_mode(): #doesnt use grad which is used fo rtrsning cuz it will speed up the output
    y_pred = model_0(xtest)

print(y_pred)
plotpredictions(predictions=y_pred)


loss_fn = nn.L1Loss()

#optimiser function(Stochastic gradient decent)

optimiser = torch.optim.SGD(params = model_0.parameters(),
                            lr=0.01)

### Building a traning loop 
# A couple of things we need in a training loop:
# 0. Loop through the data
# 1. Forward pass (this involves data moving through our model's `forward()`
#    functions) to make predictions on data - also called forward propagation
# 2. Calculate the loss (compare forward pass predictions to ground truth labels)
# 3. Optimizer zero grad
# 4. Loss backward - move backwards through the network to calculate the gradients of
#    each of the parameters of our model with respect to the loss(back propogaton)
# 5. Optimizer step - use the optimizer to adjust our model's parameters to try and
#    improve the loss(Gradient decent )


# an epochs is 1 loop through the data 
epochs = 200

#Traking didfferetn values 
epoch_count = []
loss1_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()
    #Forward pass
    y_pred = model_0(Xtrain)
 ###Caluclating the loss
    loss1 = loss_fn(y_pred , ytrain) 

   ###optimiser zero grad
    optimiser.zero_grad()
    ###perform backpropagation on the loss with respect to the parameters of the model 
    loss1.backward()
 #step the optimiser(perform gradient descent)
    optimiser.step()
   
   ###Testing 
    model_0.eval() ##truns off different settings in the model not nedde for eval
    with torch.inference_mode():
        test_pred = model_0(xtest)
        test_loss = loss_fn(test_pred,ytest)
    if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss1_values.append(loss1.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | Loss: {loss1} | testloss: {test_loss}")
            print(model_0.state_dict())


print(model_0.state_dict())
with torch.inference_mode():
    ypredsnew = model_0(xtest)
plotpredictions(predictions=ypredsnew)

plt.plot(epoch_count,loss1_values,label= "Train loss")
plt.plot(epoch_count , test_loss_values,label = "Test loss")
plt.title("Training and loss and test loss cyrve")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()


##Pythhorch save and load 
#Saving our pythorch model

MODEL_PATH = Path("model")
MODEL_PATH.mkdir(parents = True,exist_ok = True)
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
p(f"Saving model to:{MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), 
           f=MODEL_SAVE_PATH)


##loadind a model 
loaded_model_0 = linear()

p(loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH)))

loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(xtest)

p(loaded_model_preds)   

model_0.eval()
with torch.inference_mode():
    y_pred = model_0(xtest)

p(y_pred)

### Putting every thong together
##Data

