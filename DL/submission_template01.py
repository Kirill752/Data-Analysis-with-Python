import numpy as np
import torch
from torch import nn

def create_model():
 NN = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 16, bias = False),
                      nn.ReLU(),
                      nn.Linear(16, 10, bias = False),
                      nn.ReLU());
    # return model instance (None is just a placeholder)
    
 return NN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
