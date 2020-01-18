import torch
import random
from models.mlp import MLP
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn.init import kaiming_uniform_
import torch.nn as nn
import math


class MLP_Inc_Tasks(MLP):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, tasks={}, noise_type='seperate'):
        # 20 because adding
        super(MLP_Inc_Tasks, self).__init__( out_dim, in_channel, 32, hidden_dim)
        # defining seperate noise layers
        self.noise_list = nn.ModuleDict()
        self.noise_type = noise_type
        for task in tasks.keys():
            self.noise_list[task] = nn.Linear(img_sz * img_sz * in_channel, img_sz * img_sz * in_channel, bias=False)

        
    def forward(self, input, task):
        # input is the image
        input = input.view(-1,self.in_dim)
        if self.noise_type == 'seperate':
            out1_ = self.noise_list[task](input)

        else:
            # import pdb; pdb.set_trace()
            # this is the combined case
            # after training task_t the noise needs to be fixed  
            for t in range(int(task)):
                if t == 0:
                    out1_ = self.noise_list[task](input)
                else:
                    #print("====================== Augmenting task "+  str(t) + "to task" +  task + "==============")
                    out_1 = self.noise_list[task](out1_)


        out2_= super(MLP_Inc_Tasks, self).forward(out1_)

        return out2_
