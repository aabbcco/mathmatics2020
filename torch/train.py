import torch
import torch.nn as nn

class Model(nn.model):
    def __init__(self):
        super(Model, self).__init__()
        #conv layer1
        self.conv1_1 = nn.Conv2d(3, 32, 3)
        self.conv1_2 = nn.Conv2d(32, 64, 3)
        self.conv1_3 = nn.Conv2d(64, 128, 3)
        #mlp layer1
        self.mlp1_1 = nn.Conv2d(128,256, 1)
        self.mlp1_2 = nn.Conv2d(256, 128, 1)
        self.mlp1_3 = nn.Conv2d(128, 64, 1)
        #conv layer2
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 256, 3)
        self.conv2_3 = nn.Conv2d(256, 512, 3)
        