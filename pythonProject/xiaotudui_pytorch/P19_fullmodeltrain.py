import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
#定义训练的设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data=torchvision.datasets.CIFAR10('../data', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data=torchvision.datasets.CIFAR10('../data', train=False,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
train_data_size=len(train_data)
test_data_size=len(test_data)

print('训练数据集的长度为{}'.format(train_data_size))
print('测试数据集的长度为{}'.format(test_data_size))

#加载数据集
train_dataloader=DataLoader(train_data, batch_size=64)
test_dataloader=DataLoader(test_data, batch_size=64)
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
#网络模型
net=Net()
# if torch.cuda.is_available():
#     net=net.cuda()
net=net.to(device)
#损失函数
loss_function=nn.CrossEntropyLoss()
loss_function=loss_function.to(device)
#优化器
#学习率 1e-2=1*10^(-2)=0.01
learning_rate=1e-2
optimizer=torch.optim.SGD(net.parameters(), lr=learning_rate)
#设置网络训练的一些参数
#记录训练的次数
total_train_step=0
#记录测试的次数
total_test_step=0
#训练的轮数
epoch=30

writer=SummaryWriter('../log_p19')
start_time=time.time()
for i in range(epoch):
    print('---第{}轮训练开始---'.format(i+1))
    #训练步骤开始
    net.train()
    for data in train_dataloader:
        img,label=data
        # if torch.cuda.is_available():
        #     img=img.cuda()
        #     label=label.cuda()
        img=img.to(device)
        label=label.to(device)
        output=net(img)
        loss=loss_function(output,label)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step+=1
        if total_train_step%100==0:
            end_time=time.time()
            print(end_time-start_time)
            print('训练次数:{},loss:{}'.format(total_train_step,loss))
            writer.add_scalar('train_loss',loss.item(),total_train_step)
    # 测试步骤开始
    net.eval()
    test_data_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            img,label=data
            # if torch.cuda.is_available():
            #     img = img.cuda()
            #     label = label.cuda()
            img = img.to(device)
            label = label.to(device)
            output=net(img)
            loss=loss_function(output,label)
            test_data_loss+=loss.item()
            accuracy=(output.argmax(dim=1) ==label).sum()
            total_accuracy+=accuracy
    print('整体测试集的loss为{}'.format(test_data_loss))
    print('整体测试集的准确率为{}'.format(total_accuracy/test_data_size))
    writer.add_scalar('test_loss',test_data_loss,total_test_step)
    writer.add_scalar('test_loss_accuracy',total_accuracy/test_data_size,total_test_step)
    total_test_step+=1

    torch.save(net,'P19_fullmodel_{}_gpu.pth'.format(i))


writer.close()