import torchvision
from torch import nn

# train_data=torchvision.datasets.ImageNet('./data_imagenet','train',download=True,transform=torchvision.transforms.ToTensor())
vgg16=torchvision.models.vgg16(weights=True)
vgg16.classifier.add_module('add_linear', nn.Linear(1000,10 ))
print(vgg16)