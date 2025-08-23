import torch
import torchvision.transforms
from PIL import Image
from torch import nn

img_path = './数据集/dog1.png'
img=Image.open(img_path)
print(img)
transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])
img=transform(img)
# conv = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
# img=conv(img)
print(img.shape)
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
model=torch.load('P19_fullmodel_29_gpu.pth',map_location=torch.device('cpu'))
print(model)

img=torch.reshape(img,(1,3,32,32))
model.eval()
with torch.no_grad():
    output=model(img)
print(output)
print(output.argmax(1))