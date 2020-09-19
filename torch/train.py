import torch
import torch.nn as nn
from torch.autograd import Variable

NUM_epochs = 50
batch_size = 10
pre_acc=0


class NIN(nn.Module):
    def __init__(self):
        super(NIN, self).__init__()
        #conv layer1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, bias=True),
            nn.Conv2d(32, 64, 3, bias=True),
            nn.Conv2d(64, 128, 3, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding='valid')
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
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding='vaild')
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
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 128, 3),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(5)
        )
        #fc layer
        self.fc=nn.Sequential(
            nn.Linear(128* 5* 5, 128* 5, bias=True),
            nn.ReLU(),
            nn.Linear(128 * 5, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1, bias=True),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.mlp1(x)
        x = self.conv2(x)
        x = self.mlp2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

#将所有的图片resize成100*100
w = 100
h = 100
c = 3
def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist

def read_img(path):

    imgs=[]
    #label1=[]
    #label2 = []
    img_name = get_filelist(path)
    for im in img_name:
            print('reading the images:%s'% (im))
            img = cv2.imread(im)
            imgs.append(img)

    return np.asarray(imgs, np.float64)

        
if __name__ == "__main__":
    df = pd.read_csv("label1.csv")
    label1 = df["MOR"]
    label2 = df["RVR"]
    # 样本和标签的读入与分类
    path = "after_handle1"
    data = read_img(path)
    print(np.asarray(data).shape)
    label = label2,astype(np.float64)
    label = label.reshape((len(label),1))
    #打乱顺序
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    for i in range(0,len(label)):
        label[i] = math.log(label[i]+0.1, 10)


    #将所有数据分为训练集和验证集
    ratio = 0.8
    s = np.int(num_example*ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]

    #training
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    model = NIN().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    batch_per_epoch=int(x_train.shape(0) / batch_size)
    test_per_epoch = int(x_test.shape(0) / batch_size)
    for i in range(NUM_epochs):
        for i in range(batch_per_epoch):
            batch_x = Variable(x_train[i * batch_size:i * (batch_size + 1)]).to(device)
            batch_y = Variable(y_train[i * batch_size:i * (batch_size + 1)]).to(device)
            
            output = model(batch_x)
            loss = torch.nn.functional.mse_loss(batch_x, batch_y)+0.3*torch.nn.functional.l1_loss(batch_x,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        true_pos = 0
        for i in range(test_per_epoch):
            test_x = Variable(x_test[i * batch_size:i * (batch_size + 1)]).to(device)
            test_y = Variable(y_test[i * batch_size:i * (batch_size + 1)]).to(device)

            with torch.no_grad():
                test_res = model(test_x)
                for j in range(test_x.size(0)):
                    if abs(test_res[j] - test_y[j]) / test_y[j] < 0.05:
                        true_pos += 1
        acc = true_pos / (test_per_epoch * batch_size)
        if (acc > pre_acc):
            pre_acc = acc
            torch.save(model,"best_model.pth")
            

        

        