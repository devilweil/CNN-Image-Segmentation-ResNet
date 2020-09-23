# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:24:50 2020

@author: LW
"""
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision.models as models

from torchvision import datasets, transforms
import os

from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

batch_size = 8
learning_rate = 0.0002
epochs = 10

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])




def loaddata():
    train_dir = 'H:\gansu\wuwei\DataSet\SampleImage\Train'
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    
    
    val_dir = 'H:\gansu\wuwei\DataSet\SampleImage\Test'
    val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)

    
    return train_dataloader,len(train_datasets),val_dataloader,len(val_datasets)



def train():
    ###定义ResNet的网络
    model = models.resnet50(pretrained=False)
    ###导入预训练的权重，ImageNet训练来的
    model.load_state_dict(torch.load('ModelPth/resnet50-19c8e357.pth'))
    
    ####结合自己的需求，进行输出设置，最后的一层全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    ####是否进行GPU的并行
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        model.cuda()
    
    
    # feature = torch.nn.Sequential(*list(model.children())[:])
    # print(feature)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    
    Loss_list = []
    Accuracy_list = []
    train_dataloaders,datatrain_len,val_dataloaders,dataval_len=loaddata()
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_dataloaders:
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (datatrain_len), train_acc / (datatrain_len)))
        
        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in val_dataloaders:
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (dataval_len), eval_acc / (dataval_len)))
        
        Loss_list.append(eval_loss / (len(dataval_len)))
        Accuracy_list.append(100 * eval_acc / (len(dataval_len)))
    
    #模型保存
    torch.save(model, './ModelPth/model-resnet50.pth')
    
    SavePath='/content/drive/My Drive/SampleImage/LossAndAccuracy/loss.csv'
    print('Saving ',SavePath,'...')
    np.savetxt(SavePath, Loss_list, delimiter = ',')
    SavePath='/content/drive/My Drive/SampleImage/LossAndAccuracy/accuracy.csv'
    np.savetxt(SavePath, Accuracy_list, delimiter = ',')

    #loss显示
    x1 = range(0, 100)
    x2 = range(0, 100)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
        
if __name__ == "__main__": 
    # train()
    # loaddata()
    starttime = datetime.datetime.now()
    train()
    # for i in range(10):
    #     time.sleep(1)
    #     print(i)
    endtime = datetime.datetime.now()
    print('Time spend  ', endtime - starttime)
    
    
    







