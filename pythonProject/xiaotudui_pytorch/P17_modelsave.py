
import torch
import torchvision

vgg16=torchvision.models.vgg16(weights=False)
#method1 模型结构+模型参数
torch.save(vgg16,'vgg16.pth')
#method2 模型参数
torch.save(vgg16.state_dict(),'vgg161.pth')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,64,kernel_size=3,padding=1)
    def forward(self,x):
        x = self.conv1(x)
        return x
net=Net()
torch.save(net,'vgg162.pth')