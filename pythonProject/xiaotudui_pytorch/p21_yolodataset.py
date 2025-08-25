import os

import torch
from torch.utils.data import Dataset
from PIL import Image
class YOLODataset(Dataset):
    def __init__(self,img_folder,label_folder,transform,label_transform):
        self.img_folder=img_folder
        self.label_folder=label_folder
        self.transform=transform
        self.label_transform=label_transform
        self.img_names=os.listdir(self.img_folder)
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, item):
        img_name=self.img_names[item]
        #拼接文件夹和文件名
        img_path=os.path.join(self.img_folder,img_name)
        img=Image.open(img_path).convert('RGB')#转为三个通道
        label_name=img_name.split('.')[0]+".txt"
        label_path=os.path.join(self.label_folder,label_name)
        with open(label_path,'r',encoding='utf-8') as f:
            label_content=f.read()
        target=[]
        object_content=label_content.strip().split('\n')
        for object_info in object_content:
            info_list=object_info.strip().split(' ')
            class_info=float(info_list[0])
            center_x=float(info_list[1])
            center_y=float(info_list[2])
            width=float(info_list[3])
            height=float(info_list[4])
            target.extend([class_info,center_x,center_y,width,height])
        target=torch.tensor(target)
        if self.transform is not None:
            img=self.transform(img)
        return img,label_content

if __name__ == '__main__':
    tranvoc=YOLODataset(r"D:\pycharm\存储\pycharm\tudui_yolo\HelmetDataset-YOLO-Train\images",
                       r'D:\pycharm\存储\pycharm\tudui_yolo\HelmetDataset-YOLO-Train\labels',
                       transform.Compose([tr]),None)
    print(len(tranvoc))
    print(tranvoc[1])