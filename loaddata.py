import os
import random
import math
import shutil
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


def data_div(path):

    #文件存在需先移除
    if os.path.exists(path+'\\'+'test_set.txt'):
        os.remove(path+'\\'+'test_set.txt')
    if os.path.exists(path+'\\'+'train_set.txt'):
        os.remove(path+'\\'+'train_set.txt')
            
    for root_dir, sub_dirs, _ in os.walk(path):                               # 遍历os.walk(）返回的每一个三元组，内容分别放在三个变量中
        idx = 0
        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))                 # 遍历每个次级目录
            file_names = list(filter(lambda x: x.endswith('.bmp'), file_names))      # 去掉列表中的非bmp格式的文件

            random.shuffle(file_names)
            '''
            train = file_names[:int(len(file_names)*0.85)]
            test = file_names[int(len(file_names)*0.85):]
            with open('train.txt', 'w') as f1, open('test.txt', 'w') as f2:
                for i in train:
                    f1.write(i + '\n')
                for j in test:
                    f2.write(j + '\n')
            '''
            
            for i in range(len(file_names)):
                if i < math.floor(0.9 * len(file_names)):
                    txt_name = 'train_set.txt'
                elif i < len(file_names):
                    txt_name = 'test_set.txt'
                with open(os.path.join(path, txt_name), mode='a') as file:
                    file.write(os.path.join(path, sub_dir, file_names[i])+ ',' + str(idx)  + '\n') 
                    #file.write(str(idx) + ',' + os.path.join(path, sub_dir, file_names[i]) + '\n')     # 为了以后好用，修改了这里，将' '改成了','，另外路径加了sub_dir
            idx += 1
            
            '''
            for i in range(len(file_names)):
                if i < math.floor(0.8 * len(file_names)):
                    txt_name = 'train_set.txt'
                elif i < math.floor(0.9 * len(file_names)):
                    txt_name = 'val_set.txt'
                elif i < len(file_names):
                    txt_name = 'test_set.txt'
                with open(os.path.join(path, txt_name), mode='a') as file:
                    file.write(str(idx) + ',' + os.path.join(path, sub_dir, file_names[i]) + '\n')     # 为了以后好用，修改了这里，将' '改成了','，另外路径加了sub_dir
            idx += 1
            '''




# 自定义图片图片读取方式，可以自行增加resize、数据增强等操作
def MyLoader(path):
    return Image.open(path).convert('RGB')
    
class FaceDataset (Dataset):
    
    # 构造函数设置默认参数
    def __init__(self, txt, transform=None, target_transform=None, loader=MyLoader):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                line = line.rstrip( )  # 删除末尾空
                words = line.split(',')  # 以,为分隔符 将字符串分成
                imgs.append((words[0], int(words[1]))) # imgs中包含有图像路径和标签
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        #调用定义的loader方法
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
    '''
    # 构造函数设置默认参数
    def __init__(self, txt, transform=None, target_transform=None, loader=MyLoader):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                #line = line.replace(',',' ')  # 改,为空格
                words = line.split(',')  # 以空格为分隔符 将字符串分成
                imgs.append((int(words[0]),words[1]))
                #imgs.append((words[0], int(words[1]))) # imgs中包含有图像路径和标签
        self.imgs = imgs
        if transforms is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize])
        else:
            self.transform = transform

        self.target_transform = target_transform
        self.loader = loader
        

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        
        #调用定义的loader方法
        img = self.loader(fn)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
'''

if __name__ == '__main__':
    datapath=r'C:\Users\85231\Desktop\2\FaceN\datasets'
    batchsize=64
    data_div(datapath)

    train_data = FaceDataset(txt=datapath + '\\' + 'train_set.txt',transform=transforms.ToTensor())
    test_data = FaceDataset(txt=datapath + '\\' + 'test_set.txt',transform=transforms.ToTensor())
    #training_set = datasets.ImageFolder(root = "datasets")
    #train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsize)
    for data, label in enumerate(train_loader):
        print(data)
        print(label)
        break
    print('加载成功！')


    


