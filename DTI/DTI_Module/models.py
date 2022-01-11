"""
from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)


class InteractionFlat(nn.Sequential):
    '''
        Interaction Network with 2D interaction map
    '''
    
    def __init__(self, molTokenizer, proTokenizer, molEncoder, proEncoder, **config):
        super(InteractionFlat, self).__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']
        
        #densenet
        self.scale_down_ratio = config['scale_down_ratio']
        self.growth_rate = config['growth_rate']
        self.transition_rate = config['transition_rate']
        self.num_dense_blocks = config['num_dense_blocks']
        self.kernal_dense_size = config['kernal_dense_size']
        self.batch_size = config['batch_size']
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        self.gpus = torch.cuda.device_count()
        self.n_layer = 2
        #encoder
        self.molTokenizer = molTokenizer
        self.proTokenizer = proTokenizer
        self.molEncoder = molEncoder
        self.proEncoder = proEncoder
        
        self.Encoder=Encoder(self.molTokenizer,self.proTokenizer,self.molEncoder,self.proEncoder)

        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        
        self.flatten_dim = config['flat_dim'] 
        
        self.icnn = nn.Conv2d(1, 3, (50,50),stride=(4,4) , padding = 0)

        #self.icnn2 = nn.Conv2d(3, 3, (300,300), padding = 0)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            
            #output layer
            nn.Linear(32, 1)
        )

    def forward(self, df,index):
        #f = self.icnn2(self.icnn1(i_v))
        index= index.cpu().detach().numpy()
        I = self.Encoder( df, index)
        #print('Aquí2')
        #print(I.shape)
        I = torch.unsqueeze(I,1).repeat(1,1,1,1)

        #print('Aquí3')
        #print(I.shape)
        f = self.icnn(I)
    
        #print('Aquí4')
        #print(f.shape)
        f = f.view(int(self.batch_size/self.gpus), -1)
        #'''self.batch_size'''
        
        #print('Aquí5')
        #print(f.shape)
        score = self.decoder(f)
        #print("DDDD")
        return score  

class Encoder(nn.Sequential):
    
    def __init__(self, molTokenizer, proTokenizer, molEncoder, proEncoder):
        super(Encoder, self).__init__()

        self.molTokenizer = molTokenizer
        self.proTokenizer = proTokenizer
        self.molEncoder = molEncoder
        self.proEncoder = proEncoder


    def forward(self, df, index):
  
        molTokenizer = self.molTokenizer
        proTokenizer = self.proTokenizer
        molEncoder = self.molEncoder
        proEncoder = self.proEncoder

        df = df[df.index.isin(index)]
        d = df.iloc[:]['SMILES'].values.tolist()
        p = df.iloc[:]['Target Sequence'].values.tolist()
        
        cont=0
        iter=0

        for i in zip(d, p):
            try:
                inputM = molTokenizer(i[0],return_tensors = "pt")
                inputM['input_ids']=inputM['input_ids'].cuda()
                inputM['attention_mask']=inputM['attention_mask'].cuda()
         
                inputP = proTokenizer(i[1],return_tensors = "pt")
      
                inputP['input_ids']=inputP['input_ids'].cuda()
                inputP['attention_mask']=inputP['attention_mask'].cuda()
               
                outputM = molEncoder(**inputM).pooler_output
                outputP = proEncoder(**inputP).pooler_output.half().cuda()
                
                #print(iter)
                #iter +=1
                outputI = torch.outer(outputM[0], outputP[0]).half().cuda()
                outputI = outputI.view(1, 768, 768)
                  
                if  cont==0:
                  
                  I = outputI
                  cont = 1
                else:
                  
                  I = torch.cat((I,outputI),0).half().cuda()
                  
                
            except:
                  print('testing failed')
                
        
        I=I.half().cuda()

        return I 
"""
from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)


class InteractionFlat(nn.Sequential):
    '''
        Interaction Network with 2D interaction map
    '''
    
    def __init__(self, molTokenizer, proTokenizer, molEncoder, proEncoder, **config):
        super(InteractionFlat, self).__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']
        
        #densenet
        self.scale_down_ratio = config['scale_down_ratio']
        self.growth_rate = config['growth_rate']
        self.transition_rate = config['transition_rate']
        self.num_dense_blocks = config['num_dense_blocks']
        self.kernal_dense_size = config['kernal_dense_size']
        self.batch_size = config['batch_size']
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        self.gpus = torch.cuda.device_count()
        self.n_layer = 2
        #encoder
        self.molTokenizer = molTokenizer
        self.proTokenizer = proTokenizer
        self.molEncoder = molEncoder
        self.proEncoder = proEncoder
        
        self.Encoder=Encoder(self.molTokenizer,self.proTokenizer,self.molEncoder,self.proEncoder)

        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        
        self.flatten_dim = config['flat_dim'] 
        
        self.icnn = nn.Conv2d(1, 3, (3,3),stride=(4,4), padding = 0)
        #self.icnn2 = nn.Conv2d(3, 3, (300,300), padding = 0)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            
            #output layer
            nn.Linear(32, 1)
        )

    def forward(self, df,index):
        #f = self.icnn2(self.icnn1(i_v))
        index= index.cpu().detach().numpy()
        I = self.Encoder( df, index)
        #print('Aquí2')
        #print(I.shape)
        I = torch.unsqueeze(I,1).repeat(1,1,1,1)

        #print('Aquí3')
        #print(I.shape)
        f = self.icnn(I)
    
        #print('Aquí4')
        #print(f.shape)
        f = f.view(int(self.batch_size/self.gpus), -1)
        #'''self.batch_size'''
        
        #print('Aquí5')
        print(f.shape)
        score = self.decoder(f)
        #print("DDDD")
        return score  

class Encoder(nn.Sequential):
    
    def __init__(self, molTokenizer, proTokenizer, molEncoder, proEncoder):
        super(Encoder, self).__init__()

        self.molTokenizer = molTokenizer
        self.proTokenizer = proTokenizer
        self.molEncoder = molEncoder
        self.proEncoder = proEncoder


    def forward(self, df, index):
  
        molTokenizer = self.molTokenizer
        proTokenizer = self.proTokenizer
        molEncoder = self.molEncoder
        proEncoder = self.proEncoder

        df = df[df.index.isin(index)]
        d = df.iloc[:]['SMILES'].values.tolist()
        p = df.iloc[:]['Target Sequence'].values.tolist()
        
        cont=0
        iter=0

        for i in zip(d, p):
            try:
                inputM = molTokenizer(i[0],return_tensors = "pt")
                inputM['input_ids']=inputM['input_ids'].cuda()
                inputM['attention_mask']=inputM['attention_mask'].cuda()
         
                inputP = proTokenizer(i[1],return_tensors = "pt")
      
                inputP['input_ids']=inputP['input_ids'].cuda()
                inputP['attention_mask']=inputP['attention_mask'].cuda()
               
                outputM = molEncoder(**inputM).pooler_output
                outputP = proEncoder(**inputP).pooler_output.half().cuda()
                
                #print(iter)
                #iter +=1
                outputI = torch.outer(outputM[0], outputP[0]).half().cuda()
                outputI = outputI.view(1, 768, 768)
                  
                if  cont==0:
                  
                  I = outputI
                  cont = 1
                else:
                  
                  I = torch.cat((I,outputI),0).half().cuda()
                  
                
            except:
                  print('testing failed')
                
        
        I=I.half().cuda()

        return I 