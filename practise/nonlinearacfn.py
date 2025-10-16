import torch
import numpy 
from sklearn.datasets import make_circles
from torch import nn
import matplotlib.pyplot
matplotlib.use("TkAgg")   # or "Qt5Agg" if you have Qt installed
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

A = torch.arange(-10,10,1,dtype=torch.float)
print(A.dtype)

def relu(x:torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0) , x)
relu = relu(A)
print(relu)

def sigmoid(x:torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))
sig = sigmoid(A)
plt.plot(sig)
plt.show()