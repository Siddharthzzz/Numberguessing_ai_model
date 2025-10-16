import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy 
from torch import nn
import matplotlib.pyplot
matplotlib.use("TkAgg")   # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
import pandas as pd
from torchmetrics.classification import MulticlassAccuracy

#### device agnostic code  

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"uisng device: {device}")

### Setting up a traning data set 
device = "cpu"

train_data = datasets.FashionMNIST(
root="data",
train=True,
download=False,
transform=torchvision.transforms.ToTensor(),
target_transform=None
)

test_data = datasets.FashionMNIST(
root = "data",
train = False,
download=False,
transform=ToTensor(),
target_transform=None
)

class_names = train_data.classes
#classtoidx =train_data.class_to_idx
#train_data.targets

'''image,label = train_data[0]
print(f"Image Shape:{image.shape}")
plt.imshow(image.squeeze(),cmap = "gray")
plt.title(class_names[label])
plt.axis(False)
#plt.show()
'''

#plot more images 
torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows,cols = 4 , 4 
for i in range(1,rows*cols+1):
    random_idx  = torch.randint(0,len(train_data),size=[1]).item()
    img ,lbl = train_data[random_idx]
    fig.add_subplot(rows,cols,i)
    plt.imshow(img.squeeze(),cmap="gray")
    plt.title(class_names[lbl])
    plt.axis(False)
    
#plt.tight_layout()
#plt.show()
from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=False)

train_features_batch ,test_features_batch = next(iter(train_dataloader))
 
class FashionMNISTmodelV0(nn.Module):
    def __init__(self,
                 input_shape:int,
                 hidden_units:int,
                 output_shape:int
                ):
     super().__init__()
     self.layer_stack = nn.Sequential(
     nn.Flatten(),   
     nn.Linear(in_features=input_shape,out_features=hidden_units),
     nn.ReLU(),
     nn.Linear(in_features=hidden_units,out_features=hidden_units),
     nn.ReLU(),
     nn.Linear(in_features=hidden_units,out_features=hidden_units),
     nn.ReLU(),
     nn.Linear(in_features=hidden_units,out_features=output_shape),
     )
    def forward(self,x):
       return self.layer_stack(x)

class FashionMNISTmodel(nn.Module):
    def __init__(self,input_shape:int,output_shape:int,hidden_units:int):
     super().__init__()
     self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
     )
    
     self.conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
     )

     self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7,out_features=128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_features=128,out_features=output_shape),
     )

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model_cnn = FashionMNISTmodelV0(
   input_shape = 784,
   hidden_units = 128,
   output_shape = len(class_names)
)

model_cnn.to(device)
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(params=model_cnn.parameters(),lr = 0.001)

def accuracy(y_true,y_pred):
 correct = torch.eq(y_true,y_pred).sum().item()
 acc = (correct / len(y_pred))*100 
 return acc

from timeit import default_timer as timer
def print_train_time(start:float, end:float, device:torch.device = None):
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

from tqdm.auto import tqdm
torch.manual_seed(42)
torch.cuda.manual_seed(42)
train_time_start = timer()
epochs = 5
for epoch in tqdm(range(epochs)):
   print(f"Epoch: {epoch}\n------")
   train_loss = 0
  
   for batch,(X,y) in enumerate(train_dataloader):
      X, y = X.to(device), y.to(device)
      model_cnn.train()
      y_preds = model_cnn(X)
      loss = loss_fn(y_preds,y)
      train_loss += loss
      optimiser.zero_grad()
      loss.backward()
      optimiser.step()
      if batch % 400 == 0:
         print(f"batch {batch*len(X)}/{len(train_dataloader.dataset)} samples")
      
   train_loss /= len(train_dataloader)

   test_loss,testacc = 0,0
   model_cnn.eval()
   with torch.inference_mode():
      for (X_test,y_test) in test_dataloader:
         X_test, y_test = X_test.to(device), y_test.to(device)
         test_pred = model_cnn(X_test)
         test_loss += loss_fn(test_pred,y_test)
         testacc += accuracy(y_true=y_test,y_pred=test_pred.argmax(dim=1))
      test_loss /= len(test_dataloader)
      testacc /= len(test_dataloader)

   print(f"\nTrain loss {train_loss:.4f} |  test loss {test_loss:.4f},test acc: {testacc:.4f}")

train_time_end = timer()
total_train_time = print_train_time(start=train_time_start,
                                    end=train_time_end,device=str(next(model_cnn.parameters()).device))

torch.manual_seed(42)
torch.cuda.manual_seed(42)
def eval_model(model:torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               accuracy_fn):
   loss,acc =0,0
   model.eval()
   with torch.inference_mode():
      for X,y in data_loader:
         y_pred = model(X)

         loss += loss_fn(y_pred , y)
         acc += accuracy(y_true = y,y_pred = y_pred.argmax(dim=1))

      loss /= len(data_loader)
      acc /= len(data_loader)

   return {"model_name" : model.__class__.__name__,
           "model_loss" : loss.item(),
            "model_acc": acc}
        
model_cnn_results = eval_model(model=model_cnn,data_loader=test_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy)
   
print(model_cnn_results)
