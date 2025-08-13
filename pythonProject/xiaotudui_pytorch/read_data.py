from torch.utils.data import Dataset
from PIL import Image
import os
class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir =label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        #列出文件下所有文件名
        self.img_path = os.listdir(self.path)
    def __getitem__(self, item):
        img_name=self.img_path[item]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        img_label=self.label_dir
        return img,img_label
    def __len__(self):
        return len(self.img_path)
root_dir = '/Users/mac/PycharmProjects/python_code/pythonProject/xiaotudui_pytorch/数据集/hymenoptera_data/train'
label_dir = 'ants'
ants_dataset = MyDataset(root_dir,label_dir)
img,label = ants_dataset[0]
print(ants_dataset[0])
print(img.size)
print(label)