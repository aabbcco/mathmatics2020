import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import numpy as np

NUM_epochs = 50


model = vgg16(pretrained=True)
print(model)
for para in model.parameters():
    para.requires_grad = False
    
model.classifier = torch.nn.Sequential(
    nn.Linear(512* 7* 7, 4096),
    nn.LeakyReLU(),
    nn.Linear(4096, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 512),
    nn.Dropout(0.5),
    nn.Linear(512,5)
)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-6)

train_set=ImageFolder('2020_2/train', transform=ToTensor)
train_loader = DataLoader(train_set,batch_size=10,shuffle=True,num_workers=2)
test_set=ImageFolder('2020_2/test', transform=ToTensor)
test_loader = DataLoader(test_set,batch_size=10,shuffle=True,num_workers=2)


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