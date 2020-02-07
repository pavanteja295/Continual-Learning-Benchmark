import torch
import random
from types import MethodType
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn.init import kaiming_uniform_
import torch.nn as nn
import math
import models
from models.resnet import WideResNet_28_2_cifar
import torch.nn.functional as F


class Noise_Net(nn.Module):
    def __init__(self, cfg):
        # change this accordingly 
        
        super(Noise_Net, self).__init__()

        batch_size = cfg['batch_size']
        img_sz = 32 # static as we stick to CIFAr-100 max
        in_channel = 3 # using CIFAR for now
        tasks = cfg['out_dim']

        self.core_ =  models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        # redefine the logit function
        
        def n_logits(self, x):
            return x

        self.logits = self.core_.logits
        self.core_.logits =  MethodType(n_logits, self.core_)
        self.last = self.core_.last
        
        self.noise_list = {}
        # defining seperate noise layers
        for task in tasks.keys():
            self.noise_list[task] =  torch.nn.Parameter(0.2*torch.rand(batch_size, in_channel, img_sz, img_sz) +0.8 , requires_grad = True)
            #self.noise_list[task] =  torch.nn.Parameter(0.2*torch.rand(batch_size, 128) +0.8 , requires_grad = True)
            self.register_parameter('noise_list' + task, self.noise_list[task])  

    def forward(self, x, task):
        # input is the image
        if self.training:
            noise_ = self.noise_list[task]
        else:
            noise_ = torch.mean(self.noise_list[task], 0, True)

        # some times the shit loader gives only remaining samples
        x = F.relu(x * noise_[:x.shape[0]])
        out = self.core_.forward(x)
        out = self.logits(out)
        return out

    # def forward(self, x, task):
    #     # input is the image

    #     # some times the shit loader gives only remaining samples

    #     out = self.core_.forward(x)
    #     if self.training:
    #         noise_ = self.noise_list[task]
    #     else:
    #         noise_ = torch.mean(self.noise_list[task], 0, True)
        
    #     out = F.relu(out * noise_[:out.shape[0]])

    #     out = self.logits(out)

    #     return out
