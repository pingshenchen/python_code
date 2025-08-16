
from torchvision import transforms
from PIL import Image
img_path='/Users/mac/PycharmProjects/python_code/pythonProject/xiaotudui_pytorch/数据集/hymenoptera_data/train/ants/6743948_2b8c096dda.jpg'
img=Image.open(img_path)
#totensor()
tensor_img=transforms.ToTensor()(img)

#normalize
print(tensor_img[0][0][0])
trans_norm=transforms.Normalize([0.4, 0.4, 0.4], [0.5,0.5,0.5])
tensor_img_norm=trans_norm(tensor_img)
print(tensor_img_norm[0][0][0])
#resize
print(img.size)
img_resize=transforms.Resize((224,224))(img)
print(img_resize.size)
#compose
transforms_Compose=transforms.Compose([transforms.ToTensor(),trans_norm])(img)