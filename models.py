# noinspection PyUnresolvedReferences
import torch
import torch.nn as nn


rdc_text_dim = 1000
z_dim = 100
h_dim = 1024

class _param:
    def __init__(self):
        self.rdc_text_dim = rdc_text_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

class _AttributeNet(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,output_size):
        super(_AttributeNet, self).__init__()
        self.attribute = nn.Sequential(nn.Linear(input_size,h_dim),
                                     nn.LeakyReLU(),
                                     #nn.ReLU(),
                                     nn.Linear(h_dim,output_size),
                                     nn.LeakyReLU())
                                     #nn.ReLU())

    def forward(self,x):
        h = self.attribute(x)
        return h
    
class _RelationNet(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size):
        super(_RelationNet, self).__init__()
        self.relation = nn.Sequential(nn.Linear(input_size,h_dim),
                                     #nn.ReLU(),
                                     nn.LeakyReLU(),
                                     nn.Linear(h_dim,1),
                                     nn.Sigmoid()
                                     )

    def forward(self,x):
        h = self.relation(x)
        return h

