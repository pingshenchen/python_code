import torch
from P17_modelsave import *
model=torch.load("vgg16.pth")
# print(model)
model1=torch.load("vgg161.pth")
# print(model1)
model2=torch.load("vgg162.pth")
print(model2)