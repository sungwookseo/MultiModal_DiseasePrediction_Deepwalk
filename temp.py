# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:22:16 2016

@author: ace


"""

import os    
os.environ['THEANO_FLAGS'] = "device=gpu0"    
import theano

from theano import function, config, shared, sandbox
import theano.sandbox.cuda.basic_ops

theano.config.floatX = 'float32'
from keras.layers.noise import GaussianDropout
import numpy as np
from keras.layers import Input, Dense, LSTM, Embedding, MaxPooling1D, Dropout, merge
from keras.models import Model
from keras.layers.recurrent import GRU, SimpleRNN
from sklearn.preprocessing import Normalizer, StandardScaler, OneHotEncoder, Binarizer
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
from keras.layers import merge

from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,  GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score
from keras.models import model_from_json
from matplotlib import pyplot as plt# 
#from keras.regularizers import l2, activity_l2
from keras import backend as k
import matplotlib
from keras.layers import merge

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
sm = SMOTE(kind='regular')

from sklearn import preprocessing

from sklearn import cross_validation
from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold as kf
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import ShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from lifelines.utils import concordance_index as cindex
cx=[]
c1,c2=[],[]
rus = RandomUnderSampler(random_state=42)
A=[]
R=[]
p=[]
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

Z_train=[[]]*100

Z_test=[[]]*100

# C= feature 갯수
C=39 

# data preparation: 100 회 randomsampling 

for i in range(1,2):
    Z_train[i] =np.genfromtxt("output2/train/train_%s.csv"%i,delimiter=',')
    Z_test[i] =np.genfromtxt("output2/test/test_%s.csv"%i, delimiter=',')
    
for i in range(1,2):
    
    Z_train[i]=np.delete(Z_train[i],[45,46,47,48,49],1) 
    Z_test[i]= np.delete(Z_test[i], [45,46,47,48,49],1)
    #Z=Z_train[i]
    
    for v in range (1):
        #Z_train[1], Z_test[1] = train_test_split(Z,test_size=0.20, random_state=10)

        X_train=Z_train[1][:,0:C]
        y_train=Z_train[1][:,45:]
        X_train=KNN().complete(X_train)
        
        
        X_test=Z_test[1][:,0:C]
        J=np.append(X_test, X_train, axis=0)
        J=KNN().complete(J)
        X_test=J[:len(X_test[:,0]), :]
        y_test=Z_test[1][:,45:]
    
        
    
        
        n=StandardScaler()
        X_train=n.fit_transform(X_train)
        X_test=n.transform(X_test)

        colors=['r','g','b', 'y', 'c','r','g','b', 'y', 'c']
        
        # -----------  RNN  ----------------------------  
        
        
        inputs = Input(shape=(C+2,),dtype='int64')
        
        x=Embedding(input_dim=10000, output_dim=32, input_length=C+2)(inputs)
        
        
        
        x=GRU(output_dim=16, input_shape=(32,C+2), return_sequences=True)(x)
        
        x=LSTM(32)(x)
        
        
         
        x=GaussianDropout(0.1)(x)
        x = Dense(12, activation='relu')(x)
        x = Dense(8, activation='sigmoid')(x)
        
        predictions = Dense(2, activation='softmax')(x)
        
        # this creates a model that includes
        # the Input layer and three Dense layers
        model = Model(input=inputs, output=predictions)
        model.compile(optimizer='RMSprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # -------------------------------------------------------------
                      
                      
        # censored data 및  event case의 score
                      
        def score(time, event, scaling=True):
        
            seq1 = np.arange(1, len(time))
            res = np.zeros((len(time), len(time)))
        
            for i in seq1:
                for j in seq1:
                    if (time[i] > time[j] and event[j]==0) or (time[i] == time[j] and event[i] == 2 and event[j] == 0):
                        res[i,j] = 1
                    if (time[i] < time[j] and event[i] == 0) or (time[i] == time[j] and event[i] == 0 and event[j] == 2):
                        res[i,j] = -1
                    if (time[i] == time[j] and event[i] == 0 and event[j] == 0) or (time[i] < time[j] and event[i] == 2) or (event[i] == 2 and event[j] == 2):
                        res[i,j] = 0
                    if (time[i]>12.0):
                        res[i,j] = 1
                 
            temp = np.sum(res, axis=1)
        
            if scaling==True:
                temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        
            return temp
        # -----life value dimension 추가 ---------------------------------------------------------------        
        
        n=len(X_train[:,0])      
        S=np.zeros((n,1))
        S=np.insert(S, 1, [1], axis=1) 
        S0=S
        
        X_tr=X_train
            
        S=np.zeros((n,2))
        X_tr0=np.append(X_tr,S, axis=1)
    
        
        # ------model training , yearly update -----------------------------------------------------------------

        for i in range(5):
            X_tr=X_tr0
            y_tr = y_train[:,i*2:i*2+2]
            time=y_tr[:,0]
            event=y_train[:,i*2+1]
            time=np.nan_to_num(time)    
            sc=score(time, event, scaling=True) 
            y_tr1=y_tr
            censorlist=[]
            
            # followup loss case 지정
            for j in range(len(y_tr[:,0])):
                
                if event[j]==2 :
                   censorlist=np.append(censorlist,j)
            censorlist=np.asarray(censorlist, dtype='int64')  
            
            # score value ---> target
            for j in range(len(y_tr[:,0])):
                y_tr1[j,:]=[1-sc[j],sc[j]]        
            
            X_true=np.delete(X_tr, censorlist, axis=0)
            y_true=np.delete(y_tr1, censorlist, axis=0)
            
            model.fit(X_true, y_true, batch_size=30, nb_epoch=7) 
            
            # life value gradient descent 
            
            S1=model.predict(X_tr) 
            S1[np.isnan(S1)]=0
            S1=S1*100
            S1=np.floor(S1)
            S1=S1/100
            S=S+0.01*(y_tr-S1)*(S0-S1)*S1
            
            # update time (year) : S[:,0] represents 'phase'
            S[:,0]=i+1
            
           
            X_tr=np.delete(X_tr,[C,C+1], axis=1)
            X_tr0=np.append(X_tr,S,axis=1)
            
            
        #------------ model test -------------------------    
            
        n=len(X_test[:,0])      
        S=np.zeros((n,1))
        S=np.insert(S, 1, [1], axis=1) 
        S0=S
        event=[]
        time=[]
        S=np.zeros((n,2))
        S2=S
        X_tst=X_test
        
        X_tst0=np.append(X_tst, S, axis=1)
        
        # P1: for collecting survival probabilities at each year        
        P1=np.zeros((n,1)) 
        
        
        for i in range (5):
            X_tst=X_tst0
            y_tst = y_test[:,i*2:i*2+2]
        
            time=y_tst[:,0]
            event=y_test[:,i*2+1]
            time=np.nan_to_num(time)
            censorlist=[]
            sc=score(time, event, scaling=True)
            y_tst1=y_tst
            
            # remove censored data at the time interval in the test set
            for j in range(len(y_tst[:,0])):
               
                if event[j]==2 :
                   censorlist=np.append(censorlist,j)
            censorlist=np.asarray(censorlist, dtype='int64')
            
            # score of event (death) during the time interval
            for j in range(len(y_tst[:,0])):
                y_tst1[j,:]=[1-sc[j],sc[j]]        
                
                
            event=to_categorical(event)
            a= model.evaluate(X_tst, y_tst) 
            proba=model.predict(X_tst, batch_size=30)
            S1=model.predict(X_tst)
            S1[np.isnan(S1)]=0
            S1=S1*100
            S1=np.floor(S1)
            S1=S1/100
            
            # S2: previous survival provability : we used this insead of thhe true value (Y)
            
            S=S+0.01*(S2-S1)*(S0-S1)*S1
     
            S2=S
            
            # update time (year) : S[:,0] represents 'phase'
            S[:,0]=i+1 
            
            
            X_tst=np.delete(X_tst,[C,C+1], axis=1)
            X_tst0=np.append(X_tst,S,axis=1)
            
            #------- evaluate performances of the model----------------------------------------------------
            P1=np.append(P1, proba[:,1:], axis=1)
            event1=np.delete(event, censorlist, axis=0)
            proba1=np.delete(proba, censorlist, axis=0)
            
            # ROC, AUC evaluation
            fpr, tpr, thresholds = roc_curve(event1[:,1], proba1[:,1])
            roc_auc = auc(fpr, tpr)       
            plt.plot(fpr, tpr, lw=1, color=colors[i], linewidth=1)
            R=np.append(R, roc_auc)
            
            # Brier score evaluation
            a = brier_score_loss(event[:,1], proba[:,1])
            A=np.append(A,a)

        plt.xlim([0, 1])
        plt.ylim([0, 1])     
        
 
            
plt.figure(figsize=(12, 6))           
plt.subplot(1, 2, 2)


# Concordance Index evaluation    
time=y_test[:,0]      
event=y_test[:,19]
event[event==2]=1
proba=model.predict(X_tst0)
c=cindex(time, proba[:,0],event)


plt.show() 

       
