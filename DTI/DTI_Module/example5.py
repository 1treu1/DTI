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

#from models import InteractionFlat
from modelsSeq import InteractionFlat

#-------------------------------------------------------

trainAUPRC  = []
trainAUCROC  = []
ResultLoss  = []
validAUPRC  = []
validAUCROC  = []
testAUPRC  = []
testAUCROC  = []
ResultTestLoss = []
ResultValidLoss = []
logit1 = []
max1 = []

def texto():
    np . savetxt ( "trainAUPRC.txt" , trainAUPRC )
    np . savetxt ( "trainAUCROC.txt" , trainAUCROC )
    np . savetxt ( "testAUPRC.txt" , testAUPRC )
    np . savetxt ( "testAUCROC.txt" , testAUCROC )
    np . savetxt ( "validAUPRC.txt" , validAUPRC )
    np . savetxt ( "validAUCROC.txt" , validAUCROC )
    np . savetxt ( "Loss.txt" , ResultLoss )
    np . savetxt ( "testLoss.txt" , ResultTestLoss )
    np . savetxt ( "validLoss.txt" , ResultValidLoss )
    np . savetxt ( "max.txt" , max1 )

    
    #Result = np.loadtxt("") 


#-------------------------------------------------------

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
##########################################################
#Chemberta Model
Chemberta_PATH= 'seyonec/PubChem10M_SMILES_BPE_450k' #Hugging Face Smiles
molTokenizer = RobertaTokenizer.from_pretrained(Chemberta_PATH) #Selfberta Tokenizer
molEncoder = RobertaModel.from_pretrained(Chemberta_PATH) #Selfberta Model

#Paccman Model
Paccman_PATH = '~/DTI/DTI/pretrained_roberta/exp4_longformer'
PaccLarge_PATH = '/home/ubuntu/DTI/DTI/pretrained_roberta/exp4_longformer'
proTokenizer = RobertaTokenizer.from_pretrained(PaccLarge_PATH) #Fastberta Tokenizer
proEncoder = RobertaModel.from_pretrained(PaccLarge_PATH)  #Fastberta Model
#############################################################
def test(data_generator, model,df):
    
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (I,label) in enumerate(data_generator):
       
        with autocast():
              score = model.forward(df,I)
       
        #loss_fct = torch.nn.BCELoss() 
        #m = torch.nn.Sigmoid()
        #logits = torch.squeeze(m(score)) 

    
        loss_fct = torch.nn.MSELoss()
        logits = torch.squeeze(score)   
       
        label = torch.tensor(label).half().cuda()

        loss = loss_fct(logits, label)
        
        loss_accumulate += loss
        count += 1
        
        logits = logits.detach().cpu().numpy()

     
    loss = loss_accumulate/count
    
   
    
    print("Saliendo de Testing")
    
    return loss.item()
    ###############################################
    #FILE = "/content/drive/MyDrive/model/model.pth"
def main(fold_n, lr):
    config = Set_config()
    

    BATCH_SIZE = config['batch_size']
    train_epoch = 40
        
    loss_history = []
    Pre = []
    Binario = []
    model = InteractionFlat(molTokenizer, proTokenizer, molEncoder, proEncoder,**config)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model, dim = 0)


    #print("Cargando el modelo")
    #model.load_state_dict(torch.load('/content/drive/MyDrive/DTI/model'))
    #print("Modelo cargado")
            
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    #opt = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    
    print('--- Data Preparation ---')
        
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 4, 
              'drop_last': True}

   
    dataFolder = './dataset/No_binario/DAVIS/Small'
    df_train = pd.read_csv(dataFolder + '/train.csv')
    df_val = pd.read_csv(dataFolder + '/val.csv')
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
    min_loss = 0
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
            
            #loss_fct = torch.nn.BCELoss()
            #m = torch.nn.Sigmoid()
            #n = torch.squeeze(m(score))

            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score)
  
            loss = loss_fct(n, label)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if (i % 100 == 0):
                
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(loss.cpu().detach().numpy()))
                ResultLoss.append(loss.cpu().detach().numpy())
                texto()

            #print('after train')    
            #nombre=input()
          

            
        # every epoch test
        with torch.set_grad_enabled(False):
            
            loss = test(validation_generator, model,df_val)
            loss_history.append(loss)
            #loss = test(validation_generator, model,df_val)
            if loss < min_loss:
                model_max = copy.deepcopy(model)
                min_loss = loss
            
            print('Validation at Epoch '+ str(epo + 1) +  ' , Test loss: '+ str(loss))
            ResultValidLoss.append(loss)
            texto()
            #validloss.append(loss)
            #texto1()
    
    print('--- Go for Testing ---')
   

    try:
        with torch.set_grad_enabled(False):
            loss = test(testing_generator, model_max,df_test)
            #Binario, Pre, loss = test(testing_generator, model_max,df_test)
            print("Salio de test")
            print(' , Test loss: '+str(loss))
            ResultValidLoss.append(loss)
            texto()
    except Exception as e:
            #print('testing failed')
            print('testing failed: {}'.format(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            #texto3()
    return model_max, loss_history
###########################################################
import warnings
warnings.filterwarnings('ignore')

s = time()
torch.cuda.empty_cache()
model_max, loss_history = main(1, 5e-6)
e = time()
print(e-s)
lh = list(filter(lambda x: x < 1, loss_history))
plt.plot(lh)
plt.savefig('fig5.jpg')