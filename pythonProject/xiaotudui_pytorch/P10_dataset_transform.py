class CLanguage:
    # 定义__call__方法
    def __call__(self,name,add):
        print("调用__call__()方法",name,add)

clangs = CLanguage()
clangs("C语言中文网","http://c.biancheng.net")

import torchvision
tran_set = torchvision.datasets.CIFAR10(root='./downdata', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='./downdata', train=False, download=True)
img,target=test_set[0]
print(test_set.classes)
print(img)
print(target)
img.show()