import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

import os
import loaddata
from torchvision import transforms
import train
 
def main():
    # 1. load dataset
    datapath='datasets'
    batchsize = 32
    learnrate=0.05
    epochs=10
    train_data = loaddata.MyDataset(txt=datapath + '\\' + 'train_set.txt')
    test_data = loaddata.MyDataset(txt=datapath + '\\' + 'test_set.txt')
    #training_set = datasets.ImageFolder(root = "datasets")
    #train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True,num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsize,shuffle=True,num_workers=0)
    #print('加载成功！')

    # 2. load model
    net=train.MResNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    
    # 3. prepare super parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learnrate)

    # 4. train
    val_acc_list = []
    out_dir = "checkpoints/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for epoch in range(0, epochs):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            length = len(train_loader)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images) # torch.size([batch_size, num_class])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
                % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))
            
        #get the ac with testdataset in each epoch
        print('Waiting Val...')
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for batch_idx, (images, labels) in enumerate(test_loader):
                net.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Val\'s ac is: %.3f%%' % (100 * correct / total))
            
            acc_val = 100 * correct / total
            val_acc_list.append(acc_val)
 
 
        torch.save(net.state_dict(), out_dir+"last.pt")
        if acc_val == max(val_acc_list):
            torch.save(net.state_dict(), out_dir+"best.pt")
            print("save epoch {} model".format(epoch))
 
if __name__ == "__main__":
    main()