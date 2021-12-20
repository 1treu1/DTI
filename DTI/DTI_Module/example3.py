#-------------------------------------------------------
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch import nn 
from  torch.cuda.amp import autocast
import copy
import os
import sys

#-------------------------------------------------------
import transformers as tf
from transformers import RobertaTokenizer,RobertaModel


from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc
from sklearn.model_selection import KFold
torch.manual_seed(1)    # reproducible torch:2 np:3
np.random.seed(1)

from config import Set_config
from stream import Set_Data 
from models import InteractionFlat
#from modelsSeq import InteractionFlat

#-------------------------------------------------------

validloss = [] 
testloss = []
Prediccion = []
Bin = []

logit1 = []
'''
def texto1():
    np.savetxt("validloss.txt",validloss)
    
    #Result = np.loadtxt("") 
def texto2():
    np.savetxt("testloss.txt",testloss)
    np.savetxt("Pre.txt",Prediccion)
    np.savetxt("Bin.txt",Bin)
    np.savetxt("logit.txt",logit1)
    
def texto3():
    np.savetxt("testloss.txt",testloss)
    np.savetxt("validloss.txt",validloss)
'''
#-------------------------------------------------------

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
########################################################################################
#Chemberta Model
Chemberta_PATH= 'seyonec/PubChem10M_SMILES_BPE_450k' #Hugging Face Smiles
molTokenizer = RobertaTokenizer.from_pretrained(Chemberta_PATH) #Selfberta Tokenizer
molEncoder = RobertaModel.from_pretrained(Chemberta_PATH) #Selfberta Model

#Paccman Model
Paccman_PATH = '~/DTI/DTI/pretrained_roberta/exp4_longformer'
PaccLarge_PATH = '/home/ubuntu/DTI/DTI/pretrained_roberta/exp4_longformer'
proTokenizer = RobertaTokenizer.from_pretrained(PaccLarge_PATH) ###Fastberta Tokenizer
proEncoder = RobertaModel.from_pretrained(PaccLarge_PATH)  #Fastberta Model
####################################
def test(data_generator, model,df):
    
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (I,label) in enumerate(data_generator):
       
        with autocast():
              score = model.forward(df,I)
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss() 
        
            
       
        label = torch.tensor(label).half().cuda()
        #print("Tipo de dato de label")
        #print(label)

        loss = loss_fct(logits, label)
        
        loss_accumulate += loss
        count += 1
        
        logits = logits.detach().cpu().numpy()
        #print("Tipo de dato logits")
        #print(logits)
        
        #print("3")
        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        ##y_pred = y_pred//1
        #logit1.append(logits)
        #print("count", count)
        #print("y_label", label_ids.flatten().tolist() )
        #print("y_predi", logits.flatten().tolist())
     
    loss = loss_accumulate/count
    #print("Y pred")
    #print(y_pred)
    #print("Y_label")
    #print(y_label)
    #print("loss")
    #print(loss)
    
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    #print("4")
    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    #print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
    #print(y_pred_s)
    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: "+ str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    #print('Confusion Matrix : \n', cm1)
    #print('Recall : ', recall_score(y_label, y_pred_s))
    #print('Precision : ', precision_score(y_label, y_pred_s))

    total1=sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    #print('Sensitivity : ', sensitivity1 )

    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    #print('Specificity : ', specificity1)
    

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    
    #print("Saliendo de Testing")
    
    #return logits, y_pred, loss.item() #roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred, loss.item()
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred, loss.item()
##########################################################################################################
#FILE = "/content/drive/MyDrive/model/model.pth"
def main(fold_n, lr):
    config = Set_config()
    

    BATCH_SIZE = config['batch_size']
    train_epoch = 20
        
    loss_history = []
    Pre = []
    Binario = []
    model = InteractionFlat(molTokenizer, proTokenizer, molEncoder, proEncoder,**config)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model, dim = 0)


    #print("Cargando el modelo")
    #model.load_state_dict(torch.load('/home/ubuntu/DTI/DTI/model'))
    print("Modelo cargado")
            
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    #opt = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    
    print('--- Data Preparation ---')
        
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 2, 
              'drop_last': True}

   
    dataFolder = './dataset/DAVIS'
    df_train = pd.read_csv(dataFolder + '/test.csv')
    df_val = pd.read_csv(dataFolder + '/test.csv')
    df_test = pd.read_csv(dataFolder + '/test.csv')

         
    
    
    ##############################
    print("Training set")
    training_set = Set_Data(df_train.index.values, df_train.Label.values, df_train.index.values) 
    training_generator = data.DataLoader((training_set), **params)
    print("Validset")
    validation_set = Set_Data(df_val.index.values, df_val.Label.values, df_val.index.values)
    validation_generator = data.DataLoader((validation_set), **params)
    print("testing set")
    testing_set = Set_Data(df_test.index.values, df_test.Label.values, df_test.index.values)
    testing_generator = data.DataLoader(testing_set, **params)

    #for i, (I,label) in enumerate(testing_generator):
    #  print('I')
    # print(I)

    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)

    
  

    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
  
    for epo in range(train_epoch):
        torch.cuda.empty_cache()
        print('before train')
        #nombre=input()
        model.train()
        for i, (I, label) in enumerate(training_generator):
            
            with autocast():
              score =  model.forward(df_train,I)
           
            label = torch.tensor(label).half().cuda()
            
            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))
  
            loss = loss_fct(n, label)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if (i % 100 == 0):
                loss_history.append(loss)
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(loss.cpu().detach().numpy()))

            #print('after train')    
            #nombre=input()
          

            
        # every epoch test
        with torch.set_grad_enabled(False):
            
            auc, auprc, f1, logits, loss = test(validation_generator, model,df_val)
            #loss = test(validation_generator, model,df_val)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc
            
            print('Validation at Epoch '+ str(epo + 1) +  ' , Test loss: '+ str(loss))
            #validloss.append(loss)
            #texto1()
    
    print('--- Go for Testing ---')
   

    try:
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss = test(testing_generator, model_max,df_test)
            #Binario, Pre, loss = test(testing_generator, model_max,df_test)
            print("Salio de test")
            print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: '+str(f1) + ' , Test loss: '+str(loss))
            #print( ' Test loss: '+str(loss))
            print("Guardando en la lista")
            ##################################
            #Prediccion.append(Pre)
            #testloss.append(loss)
            #Bin.append(Binario)
            #texto2()
            #torch.save(model.state_dict(), FILE)
    except Exception as e:
            #print('testing failed')
            print('testing failed: {}'.format(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            #texto3()
    return model_max, loss_history
###########################################################################################
import warnings
warnings.filterwarnings('ignore')

s = time()
torch.cuda.empty_cache()
model_max, loss_history = main(1, 5e-5)
e = time()
print(e-s)
lh = list(filter(lambda x: x < 1, loss_history))
plt.plot(lh)
#######################################