#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import skopt
from skopt import gp_minimize,forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence, plot_objective,plot_evaluations,plot_histogram,plot_objective_2D
from skopt.utils import use_named_args
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K


# In[24]:


ccdb = pd.read_csv('creditcarddb/creditcard.csv')
ccdb.shape


# In[25]:


#Data was visualised in 'CreditCardFraudPrediction_DataVisualization_and_FeatureSelection' notebook
#Selecting all positives and some 15000 negatives(shuffled)
#Number of required false negatives can be tuned as well.
db1 = ccdb.loc[(np.abs(ccdb.iloc[:,12].mean() - ccdb.iloc[:,12]))/ccdb.iloc[:,12].std() > 3]
db2 = ccdb.loc[((np.abs(ccdb.iloc[:,12].mean() - ccdb.iloc[:,12]))/ccdb.iloc[:,12].std() < 3) & (ccdb['Class'] == 1) ]
db3 = ccdb.loc[((np.abs(ccdb.iloc[:,12].mean() - ccdb.iloc[:,12]))/ccdb.iloc[:,12].std() < 3) & (ccdb['Class'] == 0)]
db3 = db3.sample(frac = 1)
db3 =  db3.iloc[:15000,:]
ccdb = pd.concat([db1, db2, db3], ignore_index=True)
ccdb = ccdb.sample(frac = 1)


# In[26]:


ccdb = pd.concat([db1, db2, db3], ignore_index=True)
ccdb = ccdb.sample(frac = 1)


# In[27]:


print("Percentage of True positives in the dataset: "'{s}'" % ".format(s = round((ccdb['Class'].sum()/ccdb['Class'].count()) * 100,2)))
ccdb.shape


# In[28]:


#Due to skewness of data, f1 score will be a better measure than accuracy

def custom_f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[56]:


#This code is valid only if we use random state. train and test data must come from the same distribution.
def split_scale_data(df_x,df_y,test_size=0.2,phase='train'):
    
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state= 5) 
    Scalar = StandardScaler()
    Scalar.fit(x_train)
    x_train = Scalar.transform(x_train)
    x_test = Scalar.transform(x_test)
    
    if phase == 'train':
        x = x_train
        y = y_train
        
    elif phase == 'test':
        x = x_test
        y = y_test
        
    return x,y


# In[57]:


x_train,y_train = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1])


# In[58]:


x_train.shape


# In[32]:


#Using SKlearn's optimization library to 'kind of' automate tuning of hyperparams
#This is not the best approach as it does not provide flexibilty to tune hparams for layers individually
#In my next code 'CreditCardFraudPrediction_TensorflowAndOptuna' I use Optuna and I believe that is an amazing library to tune Hparams.

dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
dim_number_dense_layers = Integer(low=1,high=4,name='num_dense_layers')
dim_number_dense_nodes = Integer(low=32,high=512,name='num_nodes')
dim_dropout = Real(low=0,high=0.9,name='dropout')
dim_l2regular = Real(low=0,high=0.9,name='l2')
dim_batch= Integer(low=100,high=10000,name='batch')
dimensions = [dim_learning_rate, dim_number_dense_layers, dim_number_dense_nodes,dim_dropout,dim_l2regular,dim_batch]


# In[29]:


default_parameters= [1e-5,2,128,0.2,0.1,4000]


# In[30]:


def log_dir_name(learning_rate,layers,nodes,dropout,l2,batch):
    
#define path for logs    
    s="./tf_logs/lr_{0:.0e}_layers_{1}_nodes{2}_droupout{3}_l2{4}_batch{5}"
    log_dir = s.format(learning_rate,layers,nodes,dropout,l2,batch)
    
    return log_dir


# In[31]:


def create_model(learning_rate,num_dense_layers,num_nodes,dropout,l2,met=custom_f1, activation='relu'):

    model = Sequential()
    model.add(tf.keras.layers.Dense(16,input_shape = (30,),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        model.add(tf.keras.layers.Dense(num_nodes,activation=activation,name=name,kernel_regularizer=tf.keras.regularizers.l2(l2)))
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    bce = tf.keras.losses.BinaryCrossentropy()
    
    model.compile(optimizer=optimizer,loss=bce,metrics=met)
    
    return model
    


# In[32]:


path_best_model = 'best_model.keras'
best_f1 = 0.0


# In[35]:


#I used a custom metric as the fitness value, my goal was to tune Hparams to fix bias and variance at the same time

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers,num_nodes,dropout,l2,batch):
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('Number of Dense Layers: {}'.format(num_dense_layers))
    print('Number of nodes: {}'.format(num_nodes))
    print('Dropout Rate: {}'.format(dropout))
    print('L2 Reg: {}'.format(l2))
    print('Size of batch: {}'.format(batch))
    
    model = create_model(learning_rate=learning_rate, num_dense_layers=num_dense_layers, num_nodes=num_nodes,dropout=dropout,l2=l2)
    log_dir = log_dir_name(learning_rate,num_dense_layers,num_nodes,dropout,l2,batch)
    
    callback_log=TensorBoard(log_dir=log_dir,histogram_freq=1, batch_size=batch,write_graph=True,write_grads=True)
    history = model.fit(x_train,y_train,batch_size=batch,validation_split=0.2,epochs=500)
    
    f1 = history.history['val_custom_f1'][-1]
    loss = history.history['loss'][-1]
    val_loss =  history.history['val_loss'][-1]
    
    print()
    print("F1: {0:.2%}".format(f1))
    print()
    
    global best_f1
    
    if f1 > best_f1:
        model.save(path_best_model)
        best_f1=f1
        
    del model
    
    K.clear_session()
    
    return (abs(loss - val_loss)*40) - f1


# In[ ]:


#Our aim is to find Hparams that result in smallest value for fitness function
#The lowest/best fitness valuecan be -1

search_result = gp_minimize(func=fitness,dimensions=dimensions, acq_func='EI',n_calls=200,x0=default_parameters)


# # Visualize different values that Skopt used...

# In[199]:


plot_convergence(search_result)


# In[200]:


search_result.x


# In[201]:


search_result.fun


# In[202]:


sorted(zip(search_result.func_vals,search_result.x_iters))[0:99]


# In[110]:


fig =  plot_objective_2D(result=search_result,dimension_identifier1='learning_rate',dimension_identifier2='num_dense_layers')                        


# In[112]:


dim_names = ['learning_rate', 'num_dense_layers', 'num_nodes','dropout','l2','batch']
fig = plot_objective(result=search_result, dimensions=dim_names)


# In[113]:


fig = plot_evaluations(result=search_result, dimensions=dim_names)


# In[33]:


batch_size=5495


# In[34]:


#This is a little messy, A better approach will be to build a function
#Also, another way is to just load the best model instead of manually creating another model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16,input_shape = (30,),activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.04)),
    tf.keras.layers.Dropout(0.48),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.04)),
    tf.keras.layers.Dropout(0.48),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1,activation='sigmoid')
])


# In[35]:


model.summary()


# In[36]:


bce = tf.keras.losses.BinaryCrossentropy()
learning_rate=7.781316357789917e-05
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer,loss=bce,metrics=custom_f1)


# In[37]:


#F1 score is fine, but clearly there is overfitting
#I fix this in my next code using Optuina tuner tha is more flexible

history = model.fit(x_train,y_train,batch_size=batch_size,validation_split=0.1,epochs=1000)


# In[60]:


#Evaluate loss and F1 on test data

x_test,y_test = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1], phase='test')
model.test_on_batch(x_test, y_test)


# In[64]:


#Visualize loss on train and validation datasets

plt.plot(history.history['loss'][150:1000])
plt.plot(history.history['val_loss'][150:1000])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

