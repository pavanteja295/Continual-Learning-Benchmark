import torch
import random
from models.mlp import MLP
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn.init import kaiming_uniform_
import torch.nn as nn
import math


class MLP_First(MLP):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        # 20 because adding
        super(MLP_First, self).__init__( out_dim, in_channel, 32, hidden_dim)
        # defining seperate noise layers
        self.first =  nn.Linear(img_sz * img_sz * in_channel, img_sz * img_sz * in_channel, bias=False)

        
    def forward(self, input):
        # input is the image
        input = input.view(-1,self.in_dim)
        out1_ = self.first(input)
        out2_= super(MLP_First, self).forward(out1_)
        return out2_
