import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Layer5_Network(nn.Module):
    """Small architechture"""
    def __init__(self,num_classes=2):
        super(Layer5_Network, self).__init__()
        self.act=OrderedDict()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.drop_outA = nn.Dropout(0.15)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64,64,3)
        self.drop_outB = nn.Dropout(0.15)
        self.conv5 = nn.Conv2d(64,128,2)
        self.last = nn.Linear(128*4, num_classes)
    
    def logits(self, x):
        x = self.last(x)
        return x
    
    def forward(self, x):
        x = self.conv1(x)
        self.act['conv1_pre_relu']=x
        x = F.relu(x)
        x = self.conv2(x)
        self.act['conv2_pre_relu']=x
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop_outA(x)
        
        x = self.conv3(x)
        self.act['conv3_pre_relu']=x
        x = F.relu(x)
        x = self.conv4(x)
        self.act['conv4_pre_relu']=x
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop_outB(x)
        
        x = self.conv5(x)
        self.act['conv5_pre_relu']=x
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, 2)

        x = self.logits(x.view(-1, 128*4))
        # x = self.last(x)
        #self.act['fc1_output']=x
        return x
        # return F.log_softmax(x, dim=1),x
