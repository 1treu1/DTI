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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc,precision_recall_curve
from sklearn.model_selection import KFold
torch.manual_seed(1)    # reproducible torch:2 np:3
np.random.seed(1)

#----------------------------------------------------------
#----------------------------------------------------------
## MODELO 1
from .config import BIN_config_DBPE
from .models import BIN_Interaction_Flat
from .stream import BIN_Data_Encoder


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
########################################################

def test(data_generator, model):
    y_pred = [] 
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
        
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()            
        
        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)
        
        loss_accumulate += loss
        count += 1
        
        logits = logits.detach().cpu().numpy()
        
        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        
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

def main1(fold_n, lr, Api):
    config = BIN_config_DBPE() 
    
    lr = lr
    BATCH_SIZE = config['batch_size']
    train_epoch = 100
    
    loss_history = []
    logits = []
    dirModel = '/content/drive/MyDrive/DTI/Prueba_Funcionando/Modelo_1/Pruebas/P12/model/model.pth'
    model = BIN_Interaction_Flat(**config)
    model.load_state_dict(torch.load(dirModel))
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim = 0)
            
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    #opt = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    
    print('--- Data Preparation ---')
    
    params = {'batch_size': 16,
              'shuffle': False,
              'num_workers': 2, 
              'drop_last': True}
    
    df_test = pd.DataFrame.from_dict([Api]*6012)
    df_test['Label'] = 0.0
    df_test['drug_encoding'] = str('[0. 0. 0. ... 0. 0. 0.]')
    df_test['target_encoding'] = str('[ 4.469 12.214  4.071 ...  0.     0.     0.   ]')
    #df_test.append(df_test, ignore_index=True)
    #df_test.head()
    # print(df_test)
    # dataFolder = '/content/drive/MyDrive/DTI/DTI_Module/dataset/Binario/moltrans/DAVIS'
    # df_test = pd.read_csv(dataFolder + '/test.csv')
    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)
    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)
    torch.backends.cudnn.benchmark = True
    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            logits= test(testing_generator, model_max)
            ##################################
            
    except:
        print('testing failed')
    return logits


