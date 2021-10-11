#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
import optuna
from optuna.trial import TrialState


# In[79]:


#Data was visualised in 'CreditCardFraudPrediction_DataVisualization_and_FeatureSelection' notebook
#Selecting all positives and some 15000 negatives(shuffled)
#Number of required false negatives can be tuned as well.

ccdb = pd.read_csv('creditcarddb/creditcard.csv')
db1 = ccdb.loc[(np.abs(ccdb.iloc[:,12].mean() - ccdb.iloc[:,12]))/ccdb.iloc[:,12].std() > 3]
db2 = ccdb.loc[((np.abs(ccdb.iloc[:,12].mean() - ccdb.iloc[:,12]))/ccdb.iloc[:,12].std() < 3) & (ccdb['Class'] == 1) ]
db3 = ccdb.loc[((np.abs(ccdb.iloc[:,12].mean() - ccdb.iloc[:,12]))/ccdb.iloc[:,12].std() < 3) & (ccdb['Class'] == 0)]
db3 = db3.sample(frac = 1)
db3 =  db3.iloc[:15000,:]
ccdb = pd.concat([db1, db2, db3], ignore_index=True)
ccdb = ccdb.sample(frac = 1)
print("Percentage of True positives in the dataset: "'{s}'" % ".format(s = round((ccdb['Class'].sum()/ccdb['Class'].count()) * 100,2)))
ccdb.shape


# In[80]:


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


# In[81]:


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


# In[73]:


#Optuna will try to tune this using different values for H params
def define_model(trial):
    
    n_layers = trial.suggest_int("n_layers",1,3)
    layers = []
    activation_list=["relu","sigmoid"]
    
    model = tf.keras.Sequential()
    for i in range(n_layers):
        
        #We'll automate hyperparm tuning using a bayesien method
        num_nodes = int(trial.suggest_loguniform('n_units_{}'.format(i),4,128))
        activation = trial.suggest_categorical("activation_{}".format(i),activation_list)
        dropout_rate = trial.suggest_float('dropout_rate_{}'.format(i),0.0,0.95)
        lambda_reg = trial.suggest_float('lambda_reg_{}'.format(i),1e-10, 1e-3, log=True)
        
        #basic NN model 
        model.add(tf.keras.layers.Dense(num_nodes,input_shape = (30,),activation=activation,kernel_regularizer=tf.keras.regularizers.l2(lambda_reg)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.BatchNormalization())
    
    #adding the output layer
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    return model
        


# In[74]:


#We try to find the best Optimizer as well

def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float("rmsprop_learning_rate", 1e-5, 1e-1, log=True)
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float("sgd_opt_learning_rate", 1e-5, 1e-1, log=True)
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


# In[75]:


EPOCHS = 725

#Will explore Pruning more in the next project
PRUNING_INTERVAL_STEPS = 50


# In[76]:


#I used a custom metric to be as the objective, my goal was to tune Hparams to fix bias and variance at the same time

def objective(trial):
    
    x_train,y_train = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1])
    
#     learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log = True)
    
    batch_size = trial.suggest_int("batch_size",100,10000)
    optimizer = create_optimizer(trial)
    bce = tf.keras.losses.BinaryCrossentropy()
    
    # Build model and compile.
    model = define_model(trial)
    model.compile(optimizer=optimizer,loss=bce,metrics=custom_f1)
    history = model.fit(x_train,y_train,batch_size=batch_size,validation_split=0.2,epochs=EPOCHS)
    
    f1 = history.history['val_custom_f1'][-1]
    loss = history.history['loss'][-1]
    val_loss =  history.history['val_loss'][-1]

    optuna_pruning_hook = optuna.integration.TensorFlowPruningHook(
        trial=trial,
        estimator=model,
        metric=(abs(loss - val_loss)*40) - f1,
        run_every_steps=PRUNING_INTERVAL_STEPS,
    )
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    #clearing the model after every test
    del model
    K.clear_session()
    
    # :)
    return (abs(loss - val_loss)*40) - f1


# In[82]:


#This works amazingly well!
''''[I 2021-10-10 23:09:40,228] Trial 84 finished with value: -0.855547696352005 and parameters: 
{'learning_rate': 0.0021393523564096763, 'batch_size': 5326, 'optimizer': 'Adam', 'adam_learning_rate': 0.056347979238179594,
'n_layers': 2, 'n_units_0': 32.792881084659115, 'activation_0': 'relu', 'dropout_rate_0': 0.7268295637416722,
'lambda_reg_0': 5.8003894835056825e-06, 'n_units_1': 35.78671685284347, 'activation_1': 'sigmoid',
'dropout_rate_1': 0.2226073287239269, 'lambda_reg_1': 1.018643186769918e-09}. 
Best is trial 84 with value: -0.855547696352005.'''

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# In[ ]:


#I used a custom metric to be as the objective, my goal was to tune Hparams to fix bias and variance at the same time

def objective(trial):
    
    x_train,y_train = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1])
    
#     learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log = True)
    
    batch_size = trial.suggest_int("batch_size",100,10000)
    optimizer = create_optimizer(trial)
    bce = tf.keras.losses.BinaryCrossentropy()
    
    # Build model and compile.
    model = define_model(trial)
    model.compile(optimizer=optimizer,loss=bce,metrics=custom_f1)
    history = model.fit(x_train,y_train,batch_size=batch_size,validation_split=0.2,epochs=EPOCHS)
    
    f1 = history.history['val_custom_f1'][-1]
    loss = history.history['loss'][-1]
    val_loss =  history.history['val_loss'][-1]

    optuna_pruning_hook = optuna.integration.TensorFlowPruningHook(
        trial=trial,
        estimator=model,
        metric=(abs(loss - val_loss)*40) - f1,
        run_every_steps=PRUNING_INTERVAL_STEPS,
    )
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    #clearing the model after every test
    del model
    K.clear_session()
    
    # :)
    return (abs(loss - val_loss)*40) - f1


# In[85]:


#A better way would be to just load the best model instead of manually creating another model

def final_model(input_shape,lr):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32,input_shape = (input_shape,),activation='relu',kernel_regularizer=tf.keras.regularizers.l2(5.8e-06)))
    model.add(tf.keras.layers.Dropout(0.72))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(36,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(1.09e-09)))
    model.add(tf.keras.layers.Dropout(0.22))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    bce = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer,loss=bce,metrics=custom_f1)
    
    return model
    


# In[86]:


#The reason I love this tuner is because it gets of overfitting 'val loss almost equal to loss'
#F1 score is not the best, but this was expected, the source data is already PCA transformed and tough to work with
#We can further improve this by using feature selection, and changing the amount of data that we trained this model on
#F1 score of over 90 should be achievable without overfitting

model = final_model(30, 0.0563)
x_train,y_train = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1])
history = model.fit(x_train,y_train,batch_size=5326,validation_split=0.2,epochs=EPOCHS)


# In[87]:


#Evaluate loss and F1 on test data
x_test,y_test = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1], phase='test')
model.test_on_batch(x_test, y_test)


# In[89]:


#Visualize loss on train and validation datasets

plt.plot(history.history['loss'][5:750])
plt.plot(history.history['val_loss'][5:750])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

