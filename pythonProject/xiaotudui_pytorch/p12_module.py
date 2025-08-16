import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    def forward(self, input):
        output = input + input
        return output
network=Net()
input=torch.tensor(
    [[3, 7, 1, 4, 8],
     [0, 9, 5, 6, 2],
     [4, 1, 3, 7, 5],
     [8, 2, 0, 9, 6],
     [5, 4, 7, 3, 1]]
)
kernel=torch.tensor([
    [1,2,1],
    [0,2,0],
    [2,1,0]
])
print(input.shape)
print(kernel.shape)
input=torch.reshape(input,(1,1,5,5))
kernel=torch.reshape(kernel,(1,1,3,3))
print(input.shape)
print(kernel.shape)
#padding 是在周围进行弥补数据
output=F.conv2d(input,kernel,stride=1)
print(output)
