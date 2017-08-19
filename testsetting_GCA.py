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
rus = RandomUnderSampler(random_state=42)
A=[]
R=[]
p=[]
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

Z_train=[[]]*100

Z_test=[[]]*100

for i in range(1,2):
    Z_train[i-1] =np.genfromtxt("output3/train/train_%s.csv"%i,delimiter=',')
    Z_test[i-1] =np.genfromtxt("output3/test/test_%s.csv"%i, delimiter=',')
    
for i in range(0,2):
    
    Z_train[i]=np.delete(Z_train[i],[45,46,47,48,49],1) 
    Z_test[i]= np.delete(Z_test[i], [45,46,47,48,49],1)
    
    
        
    #Zt=np.append(Z_test1, Z_train, axis=0)
    #Zt=KNN().complete(Zt)
    #Z_test=Zt[:238,:]
    X_train=Z_train[i][:,0:45]
    y_train=Z_train[i][:,45:]
    X_train=KNN().complete(X_train)
    
    #Z_train=KNN().complete(Z_train)
    #Z_test=Z_test[~np.isnan(Z_test1).any(axis=1)]
    #Cen=np.append(X_train,y_train, axis=1)
    
    
    sm = SMOTE(kind='regular')
    
    X_test=Z_test[i][:,0:45]
    J=np.append(X_test, X_train, axis=0)
    J=KNN().complete(J)
    X_test=J[:len(X_test[:,0]), :]
    y_test=Z_test[i][:,45:]
    #Z_test=np.append(X_test, y_test, axis=1)
    #Z_test=Z_test[~np.isnan(Z_test).any(axis=1)]
    
    
    
    
    #sm = SMOTE(kind='regular')
    '''
    J=np.append(Xt1, X_train, axis=0)
    J=KNN().complete(J)
    Xt1=J[:131, :]
    D1=np.append(Xt1,yt, axis=1)
    D=D1[~np.isnan(D1).any(axis=1)]
    Xt=D[:,:39]
    yt=D[:,39:]
    '''

    
    
    
    
    
    #from keras.regularizers import l2, activity_l2
    
    
    n=StandardScaler()
    X_train=n.fit_transform(X_train)
    X_test=n.transform(X_test)
    X_train=X_train*100
    X_test=X_test*100
    #Xt=n.transform(Xt)
    #Xt=Xt*1000
    colors=['r','g','b','y','c']
    
    #from keras.regularizers import l2, activity_l2
    
    
    inputs = Input(shape=(47,),dtype='int64')
    
    x=Embedding(input_dim=100000, output_dim=32, input_length=47)(inputs)
    
    
    
    x=GRU(output_dim=16, input_shape=(32,47), return_sequences=True)(x)
    
    x=LSTM(32)(x)
    
    
    #auxiliary_input = Input(shape=(2,), name='aux_input')
    #x = merge([x, auxiliary_input], mode='concat')
    
    x = Dense(8, activation='relu')(x)
    
    x= Dense(32)(x)
     
    
    x = Dense(12, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    #x=GaussianDropout(0.2)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    # this creates a model that includes
    # the Input layer and three Dense layers
    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='RMSprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # censored data 처리
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
                if (time[i]< 0.0) :
                    res[i,j] =0
        temp = np.sum(res, axis=1)
    
        if scaling==True:
            temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
    
        return temp
    
    
    n=len(X_train[:,0])      
    S=np.zeros((n,1))
    S=np.insert(S, 1, [1], axis=1) 

    X_tr=X_train
    X_tr0=X_train    

    X_tr0=np.append(X_tr,S, axis=1)
    L0,L2,L4,L8,L9=[],[],[],[],[]  
    for i in range(5):
        X_tr=X_tr0
        y_tr = y_train[:,i*2:i*2+2]
        time=y_tr[:,0]
        event=y_tr[:,1]
        time=np.nan_to_num(time)    
        sc=score(time, event, scaling=True) 
        y_tr1=y_tr
        for j in range(len(y_tr[:,0])):
            y_tr1[j,:]=[1-sc[j],sc[j]]
        
        model.fit(X_tr, y_tr1, batch_size=30, nb_epoch=5) 
        S1=model.predict(X_tr) 
        S1[np.isnan(S1)]=0
        S1=S1*100*(i+1)
        S1=np.floor(S1)
        S1=S1/100
        S1=y_tr1-S1
        S=S+S1
        #S=S1
        """
        get_3rd_layer=k.function([model.layers[0].input],[model.layers[0].output])
        l0= get_3rd_layer([X_tst])[0]    
        get_3rd_layer=k.function([model.layers[0].input],[model.layers[1].output])
        l2= get_3rd_layer([X_tst])[0]
        get_3rd_layer=k.function([model.layers[0].input],[model.layers[3].output])
        l4= get_3rd_layer([X_tst])[0]
        get_3rd_layer=k.function([model.layers[0].input],[model.layers[5].output])
        l8= get_3rd_layer([X_tst])[0]
        get_3rd_layer=k.function([model.layers[0].input],[model.layers[8].output])
        l9= get_3rd_layer([X_tst])[0]   
    
        L0=np.append(L0,l0)
        L2=np.append(L2,l2)
        L4=np.append(L4,l4)
        L8=np.append(L8,l8)
        L9=np.append(L9,l9)
                
        
        """
        y_tr[y_tr==2]=0
        
        X_tr=np.delete(X_tr,[45,46], axis=1)
        X_tr=np.append(X_tr,S,axis=1)
        y_tr=np.append(y_tr, y_train, axis=1)
        y_tr=np.append(y_tr, X_tr, axis=1)
        #y_tr=y_tr[~np.isnan(y_tr).any(axis=1)]
        y_train=y_tr[:,2:22]
        X_tr0=y_tr[:,22:]
        

    '''
       
    for i in range(4,5):
        X_tr=X_train
        y_tr = y_train[:,i*2:i*2+2]
        time=y_tr[:,0]
        event=y_tr[:,1]
        time=np.nan_to_num(time)    
        sc=score(time, event, scaling=True)    
        for j in range(len(y_tr[:,0])):
            y_tr[j,:]=[1-sc[j],sc[j]]
        
        model.fit(X_tr, y_tr ,batch_size=30, nb_epoch=1) 
        S=model.predict(X_tr) 
        S[np.isnan(S)]=0
        S=S*100*(i+1)
        S=np.floor(S)
        S=S/100        
    
    '''
    n=len(X_test[:,0])      
    S=np.zeros((n,1))
    S=np.insert(S, 1, [1], axis=1) 

        
    X_tst=X_test
    
    X_test=np.append(X_tst, S, axis=1)
    P1=np.zeros((n,1))      
    for i in range (5):
        X_tst=X_test
        y_tst = y_test[:,i*2:i*2+2]
    
        time=y_tst[:,0]
        event=y_tst[:,1]
        time=np.nan_to_num(time)
        
        sc=score(time, event, scaling=True)
        y_tst1=y_tst
        for j in range(len(y_tst[:,0])):
            y_tst1[j,:]=[1-sc[j],sc[j]]
            if event[j]==2 :
                event[j]=0
            
            
        event=to_categorical(event)
        a= model.evaluate(X_tst, y_tst) 
        proba=model.predict(X_tst, batch_size=10)
        S1=model.predict(X_tst)
        S1[np.isnan(S1)]=0
        S1=S1*100*(i+1)
        S1=np.floor(S1)
        S1=S1/100
        
        S=S+S1
        #S=S1
        #y_tst[y_tst[:,1]==2]=np.nan
        #y_tst[y_tst[:,1]==0]=np.nan
        

        
        
        
        
        X_tst=np.delete(X_tst,[45,46], axis=1)
        X_tst=np.append(X_tst,S,axis=1)
        y_tst=np.append(y_tst, y_test, axis=1)
        y_tst=np.append(y_tst, X_tst, axis=1)
        #y_tst=y_tst[~np.isnan(y_tst).any(axis=1)]
        y_test=y_tst[:,2:22]
        X_test=y_tst[:,22:]
     
        A=np.append(A,a)
        P1=np.append(P1, proba[:,1:], axis=1)
        fpr, tpr, thresholds = roc_curve(event[:,1], proba[:,1])
        roc_auc = auc(fpr, tpr)       
        plt.plot(fpr, tpr, lw=1, color=colors[i], linewidth=1)
        R=np.append(R, roc_auc)
    
#L9=np.reshape(L9, (5,114,2))

plt.figure(figsize=(12, 6))           
plt.subplot(1, 2, 2)
plt.pcolor(l9,cmap=matplotlib.cm.Blues)     
      

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")

plt.show() 

       
