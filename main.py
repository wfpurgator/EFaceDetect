import loaddata
import train
import trainlib as tl
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

datapath=r'C:\Users\85231\Desktop\2\FaceN\datasets'
batchsize = 16
learnrate=0.05
epochs=10
times=1
initmodel=r'C:\Users\85231\Desktop\2\FaceN\model\Epoch9LrBz0.05-16.pth'

if __name__ == '__main__':

    train_data = loaddata.FaceDataset(txt=datapath + '\\' + 'train_set.txt',transform=transforms.ToTensor())
    test_data = loaddata.FaceDataset(txt=datapath + '\\' + 'test_set.txt',transform=transforms.ToTensor())
    #training_set = datasets.ImageFolder(root = "datasets")
    #train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True,num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsize,shuffle=True,num_workers=0)
    #print('加载成功！')

    Net= train.MResNet
    if times!=0 :
        m_state_dict = torch.load(initmodel)
        Net.load_state_dict(m_state_dict)

    trainloss,trainacc,testacc=train.train(Net,
                                           train_loader,test_loader,epochs,learnrate,batchsize,times,
                                           tl.try_gpu())
    np.set_printoptions(precision = 3)
    print('trainloss:',trainloss)
    print('trainacc:',trainacc)
    print('testacc:',testacc)