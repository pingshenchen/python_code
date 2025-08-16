import torchvision
from torch.utils.data import DataLoader
test_set = torchvision.datasets.CIFAR10(root='/Users/mac/PycharmProjects/python_code/pythonProject/xiaotudui_pytorch/数据集/downdata',train=False,transform=torchvision.transforms.ToTensor())
test_load=DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0,drop_last=False)
for data in test_load:
    img, label=data
    print(img.shape)
    print(label)