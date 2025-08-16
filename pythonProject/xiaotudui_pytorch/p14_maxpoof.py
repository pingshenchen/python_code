import torch
import torch.nn.functional as F
from torch import nn

input=torch.tensor(
    [[3, 7, 1, 4, 8],
     [0, 9, 5, 6, 2],
     [4, 1, 3, 7, 5],
     [8, 2, 0, 9, 6],
     [5, 4, 7, 3, 1]]
)
print(input.shape)
input=torch.reshape(input,(1,5,5))
print(input.shape)

class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True)
        self.sigmoid1=nn.Sigmoid()
        self.linear1=nn.Linear(25,1)
    def forward(self, x,y):
        x = self.maxpool(x)
        y=self.sigmoid1(x)
        z=self.linear1(y)
        return x,y,z
input1=torch.flatten(input)
print(input1)
net=Net()
output=net(input,input1)
print(output)