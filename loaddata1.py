import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def data_div(root):

    #文件存在需先移除
    if os.path.exists(root+'\\'+'test.txt'):
        os.remove(root+'\\'+'test.txt')
    if os.path.exists(root+'\\'+'train.txt'):
        os.remove(root+'\\'+'train.txt')
            
    #构建所有文件名的列表，dir为label
    filename = []
    #label = []
    dirs = os.listdir(root)
    for dir in dirs:
        dir_path = root + '\\' + dir
        try:
            names = os.listdir(dir_path)
        except:
            pass
        #names = os.listdir(dir_path)
        for n in names:
            filename.append(dir_path + '\\' + n + '\t' + dir)
    
    #打乱文件名列表
    np.random.shuffle(filename)
    #删除非bmp文件
    for i in range(len(filename)-1,-1,-1): # 同样不能用正序循环，for i in range(0,len(alist)), 用了remove()之后，len(alist)是动态的，会产生列表下标越界错误
        if filename[i] == 'Thumbs.db':
            filename.remove('Thumbs.db') 

    #划分训练集、测试集，默认比例4:1
    train = filename[:int(len(filename)*0.85)]
    test = filename[int(len(filename)*0.85):]
    '''
     with open(os.path.join(root, 'train.txt'), mode='a') as file1,open(os.path.join(root,'test.txt'),mode='a')as file2:
        for i in train:
            file1.write(i+'\n')
        for j in test:
            file2.write(j+'\n')
    '''
   
    #分别写入train.txt, test.txt	
    with open('train.txt', 'w') as f1, open('test.txt', 'w') as f2:
        for i in train:
            f1.write(i + '\n')
        for j in test:
            f2.write(j + '\n')
    print('成功！')


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
                line = line.rstrip()  # 删除末尾空
                words = line.split( )  # 以空格为分隔符 将字符串分成
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





if __name__ == '__main__':

    batchsize=64
    path0 = r"C:\Users\85231\Desktop\2\FaceN\datasets"
    path1= r"C:\Users\85231\Desktop\2\FaceN"

    data_div(path0)

    train_data = FaceDataset(txt=path1 + '\\' + 'train.txt', transform=transforms.ToTensor())
    test_data = FaceDataset(txt=path1 + '\\' + 'test.txt', transform=transforms.ToTensor())

    #train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsize)
    for data, label in train_loader:
        print(data.shape)
        print(label)
        break

    print('加载成功！')