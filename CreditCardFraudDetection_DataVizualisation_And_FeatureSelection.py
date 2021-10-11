#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import stats
from time import sleep
import scipy.stats
from scipy.stats import kurtosis,skew
import seaborn as sns
import abc
import SQL_connector as SQL


# In[61]:


ccdb = pd.read_csv('creditcarddb/creditcard.csv')


# In[3]:


ccdb.head(5) 


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
#to viualize how data correlates 
def threedplot(x,y,z=0,color='red',marker='x',linestyle='dashed',linewidth=0.3,markersize=0.5,height=90,rotate=45):
    fig = plt.figure(figsize = (8, 8))
    ax = plt.axes(projection = '3d')
    ax.plot3D(x,y,z, color=color, marker='x',linestyle='dashed',linewidth=0.3,markersize=0.5)
    ax.view_init(height,rotate)
    
def twodplot(x,y,color='red',marker='x',linestyle='dashed',linewidth=0.3,markersize=0.5):
    fig = plt.figure(figsize = (6.5, 6.5))
    ax = plt.axes()
    ax.plot(x,y, color=color, marker='x',linestyle='dashed',linewidth=0.3,markersize=0.5)


# In[4]:


i = 7
j = 13


# In[6]:


threedplot(ccdb.iloc[0:1000,i],ccdb.iloc[0:1000,j],color="black",height=45,rotate=-90)


# In[7]:


twodplot(ccdb.iloc[0:1000,i],ccdb.iloc[0:1000,j],color="green")


# In[13]:


#not a lot of true positives in the dataset, we need to cur down on the data or add some more synthetic data
Y = ccdb.Class
print((Y.sum()/Y.count())*100,'% fraudulent data')


# In[12]:


#to find out how many unique datapoints we have
for col_name in ccdb.columns:
    if ccdb[col_name].dtypes == 'float64':
        unique_cat = len(ccdb[col_name].unique())
        print("Feature '{col_name}' has '{unique_cat}' unique categories".format(col_name=col_name,unique_cat=unique_cat))
        


# In[13]:


#To see if we can find something intresting here, The other features are PCA transformed and we cant make much sense of it
ccdb['Amount'].value_counts().sort_values(ascending=False).head(10)


# In[29]:


ccdb.isna().sum()


# In[30]:


#Skeweness in data can be clearly seen here
ccdb['Class'].value_counts().plot(kind='bar')


# In[31]:


sns.displot(ccdb['Time'])


# In[32]:


sns.scatterplot(x='Time',y='Amount',hue='Class',data=ccdb)


# In[33]:


#correlation?
plt.figure(figsize=(28,18))
sns.heatmap(ccdb.corr(),annot=True)


# In[14]:


#outlier analysis
def percentile_outliers(features):
    uplo=[]
    for x in features:
        Q1 = np.percentile(ccdb[x], 25,interpolation = 'midpoint')
        Q3 = np.percentile(ccdb[x], 75,interpolation = 'midpoint')
        IQR=Q3-Q1
        upper = Q3 + (1.5*IQR)
        lower = Q3 - (1.5*IQR)
        labels_omitted = ccdb.loc[(ccdb[x] > upper) | (ccdb[x] < lower)]['Class'].sum()
        uplo.append(labels_omitted)
    return uplo


# In[15]:


#Most of our positives occur in outliers. Intresting.
percentile_outliers(ccdb.columns[1:-2])


# In[16]:


def normal_dist_analyser(features):
    
    for x in features:
        mean = ccdb[x].mean()
        std = ccdb[x].std()
        mini = np.floor(ccdb[x].min())
        maxi = np.ceil(ccdb[x].max())
        x_axis = np.arange(mini,maxi+1,1)
        y_axis = scipy.stats.norm.pdf(x_axis,mean,std)
        print("minimum:'{mini}', maximum:'{maxi}'".format(mini=mini,maxi=maxi))
        print("Kurtosis:'{k}'".format(k=kurtosis(ccdb[x], bias=False)))
        print("Skewness:'{s}'".format(s=skew(ccdb[x], bias=False)))
        plt.plot(x_axis,y_axis,color='coral')
        plt.title(x)
        plt.grid()
        plt.ylabel('Normal Distribution')
        
        #fill area 1
        pt1 = mean + std
        plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='none')
        pt2 = mean - std
        plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='none')
        ptx = np.linspace(pt1, pt2, 10)
        pty = scipy.stats.norm.pdf(ptx,mean,std)
        plt.fill_between(ptx, pty, color='#0b559f', alpha=1)
        
        #fill area 2
        pt1 = mean + 2*std
        plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='none')
        pt2 = mean - 2*std
        plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='none')
        ptx = np.linspace(pt1, pt2, 10)
        pty = scipy.stats.norm.pdf(ptx,mean,std)
        plt.fill_between(ptx, pty, color='#2b6bba', alpha=0.75)
        
        #fill area 3
        pt1 = mean + 3*std
        plt.plot([pt1 ,pt1 ],[0.0,scipy.stats.norm.pdf(pt1 ,mean, std)], color='none')
        pt2 = mean - 3*std
        plt.plot([pt2 ,pt2 ],[0.0,scipy.stats.norm.pdf(pt2 ,mean, std)], color='none')
        ptx = np.linspace(pt1, pt2, 10)
        pty = scipy.stats.norm.pdf(ptx,mean,std)
        plt.fill_between(ptx, pty, color='#2b72ba', alpha=0.5)
        
        plt.show()


# In[17]:


#To determine if I can go ahead and use Z outliers, need to make sure data is kinda normally distributed. given the volume of samples Central limit Theorem should work in our favour here
#postive kurtosis = heavy tails (more outliers)
normal_dist_analyser(ccdb.columns[1:-2])


# In[19]:


def Z_outliers(features):
    uplo=[]
    for x in features:
        labels_omitted = ccdb.loc[(np.abs(ccdb[x].mean() - ccdb[x]))/ccdb[x].std()>3]['Class'].sum()
        uplo.append(labels_omitted)
    return uplo


# In[26]:


#This confirms that we cant just get rid of outliers, infact, most of the true positives lie within outliers.
Z_outliers(ccdb.columns[1:-2])


# In[27]:


#More visualisation
sns.kdeplot(ccdb.V12,shade=True)


# In[28]:


sns.kdeplot(ccdb.V25,ccdb.V23,shade=True,shade_lowest=False,cbar=True)


# # I experimented with different feature selection techniques below, but I dont use it on my final dataset; I found a better way.

# In[62]:


def split_scale_data(df_x,df_y,test_size=0.2):
    
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2) 
    Scalar = StandardScaler()
    Scalar.fit(x_train)
    x_train = Scalar.transform(x_train)
    x_test = Scalar.transform(x_test)
    
    return x_train,y_train


# In[63]:


#To explore feature importance (ANOVA/F-score)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

x_train,y_train = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1])

selector = SelectKBest(f_classif, k=20)
selector.fit(x_train,y_train)

selector.scores_

cols = selector.get_support(indices=True)
cols


# In[64]:


ccdb = ccdb.iloc[:,list(cols)]
ccdb


# In[34]:


#Input is numerical, hence its not wise to use Mutual info or Chi Square. I wont use this blocks output for analysis
#To explore feature importance (mutual)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif

x_train,y_train = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1])

selector = SelectKBest(score_func=mutual_info_classif, k=27)
selector.fit(x_train,y_train)

selector.scores_

cols = selector.get_support(indices=True)
cols


# In[35]:


selector.scores_


# In[37]:


#Using other methods to find important features 
#I dont really use this
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import accuracy_score


x_train,y_train = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1])

sel = SelectFromModel(DecisionTreeClassifier())
sel.fit(x_train,y_train)
sel.get_support()


# In[38]:


x_train.columns[sel.get_support()]


# In[39]:


dt = DecisionTreeClassifier()
rfcev = RFECV(estimator=dt,step=1,scoring='accuracy')
rfcev.fit(x_train,y_train)
sel.get_support()


# In[40]:


x_train.columns[sel.get_support()]


# In[42]:


#PCA for feature reduction.
#I dont use this either.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[47]:


x_train,y_train = split_scale_data(ccdb.iloc[:,:-1],ccdb.iloc[:,-1])

#Transform data bedore PCA
scalar =  StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

#Apply PCA
pca = PCA(n_components=30)
pca.fit(x_train)

x_train = pca.transform(x_train)
x_test = pca.transform(x_train)


# In[235]:


#I was testing my SQL database on GCP this is not required, I left this code here for my reference
import sqlite3
import mysql.connector
import getpass

class sqllogin:
    
    def __init__(self):
        self.password = getpass.getpass(prompt='Password: ', stream=None)
        self.connection(self.password)

    def connection(self,password):
        self.conn = mysql.connector.connect(user='xyz', password=password, host='external ip here', database='Features')
        self.cursor = self.conn.cursor(buffered=True)
        return self.cursor
    
sql = sqllogin()
    


# In[306]:


class queries:
    
    def __init__(self,ccdb):
        self.debug = False
        self.ccdb = ccdb
        self.getLabels(self.ccdb)
    
    def dropTable(self,table):
        q = (("".join(['DROP TABLE ',table, ';'])))
        sql.cursor.execute(q)
        
        if self.debug == True: print(q)
            
    def createTable(self,table):
        q = (("".join(['CREATE TABLE ',table, '(ID int NOT NULL AUTO_INCREMENT, PRIMARY KEY(ID));'])))
        sql.cursor.execute(q)
        
        if self.debug == True: print(q)
        
    def modifyTable(self,ccdb,table):
        for column in ccdb.columns:
            q = (("".join(['ALTER TABLE ',table,' ADD ',column,' float(7);'])))
            sql.cursor.execute(q)
            
            if self.debug == True: print(q)
                
    def getLabels(self,ccdb):
        conc = []
        for column in ccdb.columns:
            conc.append(column)
            
        self.labels = (str(conc).replace('[','').replace(']','').replace(' ','').replace("'",""))
        return self.labels
    
    def insert(self,ccdb,table,labels):
        for i in range(0,100):
            s = str(list(ccdb.iloc[i,:].to_numpy().astype('float16'))).replace('[','').replace(']','')
            q = "".join(['INSERT INTO ', table, ' (',labels,') VALUES ','(',s,');'])
            sql.cursor.execute(q)
            
            if self.debug == True: print(q)
            
q = queries(ccdb)


# In[307]:


table = 'ANOVA'
q.debug=True
q.dropTable(table)
q.createTable(table)
q.modifyTable(q.ccdb,table)
q.insert(q.ccdb,table,q.labels)

