import torch
import torch.nn as nn

class Model(nn.model):
    def __init__(self):
        super(Model, self).__init__()
        #conv layer1
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        #mlp layer1
        self.mlp1_1 = nn.Conv2d(128,256, 1)
        self.mlp1_2 = nn.Conv2d(256, 128, 1)
        self.mlp1_3 = nn.Conv2d(128, 64, 1)
        