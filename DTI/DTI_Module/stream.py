import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json


class Set_Data(data.Dataset):

    def __init__(self, list_IDs, labels,I):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.I = I
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        I = self.I[index]     
        y = self.labels[index]
        
        return I,y