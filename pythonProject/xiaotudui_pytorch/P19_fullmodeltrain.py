import torchvision
from p15_nnseq import *
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

#网络模型
net=Net()
#损失函数
loss_function=nn.CrossEntropyLoss()
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
epoch=10
for i in range(epoch):
    print('---第{}轮训练开始---'.format(i+1))
    for data in train_dataloader:
        img,label=data
        output=net(img)
        loss=loss_function(output,label)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step+=1
        if total_train_step%100==0:
            print('训练次数:{},loss:{}'.format(total_train_step,loss))
    test_data_loss=0
    with torch.no_grad():
        for data in test_dataloader:
            img,label=data
            output=net(img)
            loss=loss_function(output,label)
            test_data_loss+=loss.item()
    print('整体测试集的loss为{}'.format(test_data_loss))