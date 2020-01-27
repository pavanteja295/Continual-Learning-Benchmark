import torch
import random
from models.mlp import MLP
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn.init import kaiming_uniform_
import torch.nn as nn
import math
from models.resnet import PreActBlock, PreActResNet_cifar

class Noise_Layer(nn.Module):
    def __init__(self, in_channel=1, img_sz=32, op='*'):
        super(Noise_Layer, self).__init__()
        self.weight = nn.Parameter(torch.randn(int(in_channel), int(img_sz), int(img_sz) ))
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

class Noise_Net(PreActResNet_cifar):
    def __init__(self, out_dim=10, in_channel=3, img_sz=32, hidden_dim=256, tasks={}, noise_type='seperate'):
        # change this accordingly 
        super(Noise_Net, self).__init__(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim)
        self.noise_list = nn.ModuleDict()
        self.noise_type = noise_type
        # defining seperate noise layers
        for task in tasks.keys():
            self.noise_list[task] = Noise_Layer(in_channel, img_sz)
        
        
    def forward(self, input, task):
        # input is the image
        
        # input = input.view(-1,self.in_dim)
        if self.noise_type == 'seperate':
            out1_ = self.noise_list[task](input)

        else:
            if self.training:
                # this is the combined case
                # after training task_t the noise needs to be fixed  
                for t in range(1, int(task) + 1):
                    if t == 1:
                        out1_ = self.noise_list[str(t)](input)
                    else:
                        # print("====================== Augmenting task "+  str(t) + "to task" +  task + "==============")
                        out_1 = self.noise_list[str(t)](out1_)

            else:
                # task not used
            
                for t in range(1, len(self.noise_list.keys()) + 1):
                    if t == 1:
                        out1_ = self.noise_list[str(t)](input)
                    else:
                        # print("====================== Evaluating task "+  str(t) + "to task" +  task + "==============")
                        out_1 = self.noise_list[str(t)](out1_)

        out2_= super(Noise_Net, self).forward(out1_)

        return out2_
