# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:07:47 2020

@author: Aamod Save
"""

'''
############################################
#          INITIALIZING LIBRARIES          #
############################################
'''

#Importing the libraries
from torch import nn
import torch.nn.functional as F
import torch


'''
#######################################
#          DEFINE THE  MODEL          #
#######################################
'''

class Network(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
                
        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.out = nn.Linear(128, self.output_dim)
    
    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x
            
        
        