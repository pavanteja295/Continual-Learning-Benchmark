import torch
import random
from models.mlp import MLP
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn.init import kaiming_uniform_
import torch.nn as nn
import math

class Noise_Layer(nn.Module):
    def __init__(self, in_channel=1, img_sz=32, op='*'):
        super(Noise_Layer, self).__init__()
        self.weight = nn.Parameter(torch.randn(int(img_sz * img_sz * in_channel)))
        # self.reset_parameters()
        self.op = op

    # def reset_parameters(self):
    #     kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, in_):
        if self.op == '*':
            out_ = in_ * self.weight
        elif self.op == '+':
            out_ = in_ + self.weight
        return out_

class Noise_Net(MLP):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, tasks={}, noise_type='seperate'):
        # import pdb; pdb.set_trace()
        super(Noise_Net, self).__init__( out_dim, in_channel, img_sz, hidden_dim)
        self.noise_list = nn.ModuleDict()
        self.noise_type = noise_type
        # defining seperate noise layers
        for task in tasks.keys():
            self.noise_list[task] = Noise_Layer(in_channel, img_sz)

        
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
                    out_1 = self.noise_list[task](out1_)


        out2_= super(Noise_Net, self).forward(out1_)

        return out2_
