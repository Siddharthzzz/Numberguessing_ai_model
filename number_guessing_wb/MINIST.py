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
from torchvision.transforms import v2
###setting up device agnostic code 
from torch.utils.data import DataLoader


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

###Defining a transformer 
transform = v2.Compose([
v2.ToImage(),
v2.ToDtype(torch.float32, scale=True),
v2.Normalize((0.1307,),(0.3081,))
] )

train_data = datasets.MNIST(
root="data",
train=True,
download=False,
transform=transform,
target_transform = None
)

test_data = datasets.MNIST(
root = "data",
train = False,
download = False,
transform = transform,
target_transform=None
)

class_names = train_data.classes

BATCH_SIZE = 64

train_loader = DataLoader(dataset=train_data ,batch_size=BATCH_SIZE ,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=True)

images , labels = next(iter(train_loader))

class MNISTmodel(nn.Module):
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

    def forward(self,x):
       x = self.conv_block1(x)
       x = self.conv_block2(x)
       x = self.classifier(x)
       return x
   
Numberguessing_model = MNISTmodel(input_shape=1,output_shape=10,hidden_units=128)
Numberguessing_model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(Numberguessing_model.parameters(),lr=0.001)

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
train_time_start = timer()
epochs = 1
for epoch in tqdm(range(epochs)):
   print(f"Epoch: {epoch}\n------")
   training_loss = 0

   for batch,(images,labels) in enumerate(train_loader):
      images,labels=images.to(device),labels.to(device)
      Numberguessing_model.train()
      labels_pred = Numberguessing_model(images)
      loss = loss_fn(labels_pred,labels)
      training_loss += loss
      optimiser.zero_grad()
      loss.backward()
      optimiser.step()
      if batch % 400 == 0:
         print(f"batch {batch*len(images)}/{len(train_loader.dataset)} samples")

   training_loss /= len(train_loader)
   test_loss , test_acc = 0 ,0
   Numberguessing_model.eval()
   with torch.inference_mode():
       for (images_test,labels_test) in test_loader:
          images_test, labels_test = images_test.to(device), labels_test.to(device)
          test_pred = Numberguessing_model(images_test)
          test_loss += loss_fn(test_pred,labels_test)
          test_acc += accuracy(y_true= labels_test,y_pred=test_pred.argmax(dim=1))
       test_loss /= len(test_loader)
       test_acc /= len(test_loader)
       print(f"\nTrain loss {training_loss:.4f} |  test loss {test_loss:.4f},test acc: {test_acc:.4f}")

train_time_end = timer()
total_train_time = print_train_time(start=train_time_start,
                                    end=train_time_end,device=str(next(Numberguessing_model.parameters()).device))


def eval_model(model:torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               accuracy_fn):
   loss,acc =0,0
   model.eval()
   with torch.inference_mode():
      for X,y in data_loader:
         X,y = X.to(device),y.to(device)
         y_pred = model(X)
         
         loss += loss_fn(y_pred , y)
         acc += accuracy(y_true = y,y_pred = y_pred.argmax(dim=1))

      loss /= len(data_loader)
      acc /= len(data_loader)

   return {"model_name" : model.__class__.__name__,
           "model_loss" : loss.item(),
            "model_acc": acc}
        
Numberguessing_model_results= eval_model(model=Numberguessing_model,data_loader=test_loader,loss_fn=loss_fn,accuracy_fn=accuracy)
   
print(Numberguessing_model_results)




