import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import cv2
import pandas as pd
import os
import numpy as np
import math

NUM_epochs = 100
batch_size = 10
pre_acc=0

class custom_set(Dataset):
    def __init__(self,datapath,labelpath):
        self.data = np.load(datapath)
        self.label = np.load(labelpath)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        img = self.data[idx]
        label =self.label[idx]
        return img,label

class NIN(nn.Module):
    def __init__(self):
        super(NIN, self).__init__()
        #conv layer1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, bias=True),
            nn.Conv2d(32, 64, 3, bias=True),
            nn.Conv2d(64, 128, 3, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        #mlp layer1
        self.mlp1 = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64,1e-5,0.9)
        )
        #conv layer2
        self.conv2=nn.Sequential(
            nn.Conv2d(64, 128, 3, bias=True),
            nn.Conv2d(128, 256, 3, bias=True),
            nn.Conv2d(256, 512, 3, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        #mlp layer2
        self.mlp2=nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64, 1e-5,0.9)
        )
        #conv layer3
        self.conv3=nn.Sequential(
            nn.Conv2d(64, 128, 3, bias=True),
            nn.Conv2d(128, 256, 3, bias=True),
            nn.Conv2d(256, 512, 3, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.mlp3=nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64, 1e-5,0.9)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(64, 128, 3, bias=True),
            nn.Conv2d(128, 256, 3, bias=True),
            nn.Conv2d(256, 256, 3, bias=True),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(5)
        )
        #fc layer
        self.fc=nn.Sequential(
            nn.Linear(256* 5* 5, 256* 5, bias=True),
            nn.ReLU(),
            nn.Linear(256 * 5,256, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6, bias=True),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.mlp1(x)
        x = self.conv2(x)
        x = self.mlp2(x)
        x = self.conv3(x)
        x = self.mlp3(x)
        x=self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output



        
if __name__ == "__main__":

    train = custom_set("x_train.npy","y_train.npy")
    test = custom_set("x_val.npy","y_val.npy")
    train_loader = DataLoader(dataset=train,batch_size=10,shuffle=True,num_workers=4)
    test_loader =DataLoader(dataset=test,batch_size=10,shuffle=True,num_workers=4)


    #training
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    model = NIN().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    for i in range(NUM_epochs):
        print("epoch: ", i)
        for i,traindata in enumerate(train_loader):
            batch_x,batch_y =traindata
            batch_x = (batch_x.type(torch.FloatTensor)).to(device)
            batch_y=(batch_y.type(torch.LongTensor)).to(device)
            output = model(batch_x)
            #loss = torch.nn.functional.cross_entropy(torch.mea)
            loss = torch.nn.functional.cross_entropy(output,batch_y,reduction='mean')
            if not i%5:
                print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        true_pos = 0
        for i,testdata in enumerate(test_loader):
            test_x,test_y = testdata
            test_x = (test_x.type(torch.FloatTensor)).to(device)
            test_y=(test_y.type(torch.LongTensor)).to(device)
            with torch.no_grad():
                test_res = model(test_x)
                true_pos+=torch.sum(torch.argmax(test_res)==test_y)
        acc = true_pos/test.data.shape[0]
        print("acc: ",acc)
        if (acc > pre_acc):
            pre_acc = acc
            torch.save(model,"best_model.pth")
            
