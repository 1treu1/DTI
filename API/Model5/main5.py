#%cd /content/drive/MyDrive/DTI/Prueba_Funcionando/Modelo_2
#-------------------------------------------------------
#-------------------------------------------------------
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

from  torch.cuda.amp import autocast
from torch import nn 
import copy

#--------------------------------------------------------
import transformers as tf
from transformers import RobertaTokenizer,RobertaModel
#--------------------------------------------------------
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc
from sklearn.model_selection import KFold
torch.manual_seed(1)    # reproducible torch:2 np:3
np.random.seed(1)

#----------------------------------------------------------
#----------------------------------------------------------

from .stream import BIN_Data_Encoder
from .stream import Set_Data
from .config import BIN_config_DBPE
from .models5 import Interaction_Module


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

n_epoch = 0

#----------------------------------------------------------

#------------------------------------------------------------

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)

from torch.utils import data
import json

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs

def test(data_generator, model,df):
    
      y_pred = []
      y_label = []
      model.eval()
      loss_accumulate = 0.0
      count = 0.0
      for i, (d, p, p_mask,label) in enumerate(data_generator):
            
                with autocast():    
                      score = model.forward(d, p.long().cuda(), p_mask.long().cuda())
                
                m = torch.nn.Sigmoid()
                logits = torch.squeeze(m(score))
                loss_fct = torch.nn.BCELoss()            
                
                label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

                loss = loss_fct(logits, label.half().cuda())
                
                loss_accumulate += loss
                count += 1
                
                logits = logits.detach().cpu().numpy()
                
                label_ids = label.to('cpu').numpy()
                y_label = y_label + label_ids.flatten().tolist()
                y_pred = y_pred + logits.flatten().tolist()
            

      print('CUDA MEMORY USED 4:',torch.cuda.memory_allocated() )     
      loss = loss_accumulate/count
      
      fpr, tpr, thresholds = roc_curve(y_label, y_pred)

      precision = tpr / (tpr + fpr)

      f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

      thred_optim = thresholds[5:][np.argmax(f1[5:])]

      print("optimal threshold: " + str(thred_optim))

      y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

      auc_k = auc(fpr, tpr)
      print("AUROC:" + str(auc_k))
      print("AUPRC: "+ str(average_precision_score(y_label, y_pred)))

      cm1 = confusion_matrix(y_label, y_pred_s)
      print('Confusion Matrix : \n', cm1)
      print('Recall : ', recall_score(y_label, y_pred_s))
      print('Precision : ', precision_score(y_label, y_pred_s))

      total1=sum(sum(cm1))
      #####from confusion matrix calculate accuracy
      accuracy1=(cm1[0,0]+cm1[1,1])/total1
      print ('Accuracy : ', accuracy1)

      sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
      print('Sensitivity : ', sensitivity1 )

      specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
      print('Specificity : ', specificity1)

      outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    
      return y_pred_s


def main5(fold_n, lr, Api):
    config = BIN_config_DBPE()
    
    lr = lr
    BATCH_SIZE = config['batch_size']
    train_epoch = 20
    
    loss_history = []
    
    #Fullmodel
    model=Interaction_Module(**config)
    
    FILE = "/content/drive/MyDrive/DTI/Prueba_Funcionando/Modelo_5/Pruebas/model5.pth"
    model.load_state_dict(torch.load(FILE))
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim = 0)
            
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    #opt = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    
    print('--- Data Preparation ---')
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6, 
              'drop_last': True}

    df_test = pd.DataFrame.from_dict([Api]*6012)
    df_test['Label'] = 0.0
    df_test['drug_encoding'] = str('[0. 0. 0. ... 0. 0. 0.]')
    df_test['target_encoding'] = str('[ 4.469 12.214  4.071 ...  0.     0.     0.   ]')
    

    # dataFolder = '/content/drive/MyDrive/DTI/DTI_Module/dataset/Binario/moltrans/DAVIS'
    # df_train = pd.read_csv(dataFolder + '/train.csv')
    # df_val = pd.read_csv(dataFolder + '/val.csv')
    # df_test = pd.read_csv(dataFolder + '/test.csv')
    
     ##############################
    print("testing set")
    test_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    test_generator = data.DataLoader(test_set, **params)

    # early stopping
    max_auc = 0
    max_pr = 0
    model_max = copy.deepcopy(model)
    
    
    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            logits= test(test_generator, model_max, df_test)

    except:
        print('testing failed')
    return logits

