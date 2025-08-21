import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10(root='/Users/mac/PycharmProjects/python_code/pythonProject/xiaotudui_pytorch/数据集/downdata',train=False,transform=torchvision.transforms.ToTensor())
dataload=DataLoader(dataset=test_set,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model=nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),

            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )
    def forward(self,x):
        x = self.model(x)
        return x
# net=Net()
# input = torch.ones((64,3,32,32))
# output = net(input)
# print(output.size())
# loss=nn.CrossEntropyLoss()
# optim=torch.optim.SGD(net.parameters(),lr=0.01)
# for epoch in range(20):
#     running_loss=0.0
#     for data in dataload:
#         img,targets=data
#         output=net(img)
#         result=loss(output,targets)
#         optim.zero_grad()
#         result.backward()
#         optim.step()
#         running_loss +=result
#     print(running_loss)
