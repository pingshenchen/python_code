import torch
import torchvision
from torch.nn import Module, Conv2d

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root='/Users/mac/PycharmProjects/python_code/pythonProject/xiaotudui_pytorch/数据集/downdata',train=False,transform=torchvision.transforms.ToTensor())
dataload=DataLoader(dataset=test_set,batch_size=64)

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=Conv2d(in_channels=3, out_channels=6, kernel_size=3,padding=3)
    def forward(self, x):
        x = self.conv1(x)
        return x
writer=SummaryWriter(log_dir='../logs')
net=Net()
print(net)
step=0
for data in dataload:
    img,label=data
    output=net(img)
    print(img.shape)
    print(output.shape)
    writer.add_images('input',img,global_step=step)
    output=torch.reshape(output,(-1,3,36,36))
    writer.add_images('output',output,global_step=step)
    step=step+1