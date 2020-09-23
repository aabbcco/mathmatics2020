import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
import  torchvision.transforms as trans
import numpy as np
import tensorboardX

NUM_epochs = 100


model = vgg16(pretrained=True)

for para in model.parameters():
    para.requires_grad = False
    
model.classifier = torch.nn.Sequential(
    nn.Linear(512* 7* 7, 4096),
    nn.LeakyReLU(),
    nn.Linear(4096, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 512),
    nn.Dropout(0.5),
    nn.Linear(512,5),
    nn.Softmax(dim=1)
)
print(model)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = model.to(device)

transform = trans.Compose(
    [trans.RandomHorizontalFlip(),
    trans.ToTensor()]
)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-6)

train_set=ImageFolder('2020_2/train',transform=transform)
train_loader = DataLoader(train_set,batch_size=20,shuffle=True,num_workers=2)
test_set=ImageFolder('2020_2/test',transform=transform)
test_loader = DataLoader(test_set,batch_size=10,shuffle=True,num_workers=2)

writer  = tensorboardX.SummaryWriter(log_dir='log')

pre_acc =0
global_step = 0
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
        writer.add_scalar('loss/loss',loss,global_step=global_step )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step+=1
    
    true_pos = 0
    num_test =0
    for i,testdata in enumerate(test_loader):
        test_x,test_y = testdata
        test_x = (test_x.type(torch.FloatTensor)).to(device)
        test_y=(test_y.type(torch.LongTensor)).to(device)
        with torch.no_grad():
            test_res = model(test_x)
            true_pos+=(torch.argmax(test_res,1)==test_y).sum().float()
            num_test+=test_x.shape[0]
    acc = true_pos/num_test
    print("acc: ",acc)
    writer.add_scalar('acc/acc',acc,global_step=global_step )
    if (acc > pre_acc):
        pre_acc = acc
        torch.save(model,"best_model.pth")

#output all test_data
#restore the best model
# chek = torch.load('best_model.pth')
# model.load_state_dict(chek['net'])

for i,testdata in enumerate(test_loader):
    test_xs,test_ys = testdata
    test_x = (test_xs.type(torch.FloatTensor)).to(device)
    test_y=(test_ys.type(torch.LongTensor)).to(device)
    with torch.no_grad():
        test_res = model(test_x)
        y_pred =torch.argmax(test_res,1).to('cpu').numpy().astype(int)
        true_pos+=(torch.argmax(test_res,1)==test_y).sum().float()
        num_test+=test_x.shape[0]
        for k,gt in enumerate(test_ys):
            np.save('results/'+str(i)+'_'+str(k)+'_gt_'+str(int(test_ys[k]))+'_pred_'+str(int(y_pred[k]))+'.npy',test_xs[k].numpy().astype(float))

acc = true_pos/num_test
print("final acc ! : ",acc)
