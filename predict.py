import train
import torch
import loaddata
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import trainlib as tl
import numpy as np
import matplotlib.pyplot as plt

datapath=r'C:\Users\85231\Desktop\2\FaceN\datasets'
initmodel=r'C:\Users\85231\Desktop\2\FaceN\model\Epoch19Lr0.05Bz16.pth'
batchsize = 16
#数量
n=10
#绘制
x=2
y=5


if __name__ == '__main__':

    #读测试集（选图）
    test_data = loaddata.FaceDataset(txt=datapath + '\\' + 'test_set.txt',transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=batchsize,shuffle=True,num_workers=0)
    #读网络
    net= train.MResNet
    m_state_dict = torch.load(initmodel)
    net.load_state_dict(m_state_dict)
    net=net.to('cpu')
    #n 预测
    for PIC, table in test_loader:
        break
    trues = table.numpy()
    preds = tl.argmax(net(PIC), axis=1)
    preds=preds.numpy()
    #绘制
    for i in range(n):
        plt.subplot(x,y,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title('T:'+str(trues[i])+'P:'+str(preds[i]))
        img = transforms.ToPILImage()(PIC[i])
        plt.imshow(img)
    plt.suptitle('True/Predict')
    plt.show()

'''  
    for images, _, _, _, _ in trainloader:
        for i in range(args.train_batch):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            plt.subplot(1, args.train_batch, i + 1)
            # plt.xticks([])  # 去掉x轴的刻度
            # plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()
'''

