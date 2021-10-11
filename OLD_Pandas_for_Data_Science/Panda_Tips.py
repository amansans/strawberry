#!/usr/bin/env python
# coding: utf-8

# In[17]:


#in thos project we will read a CSV file containing information about cutomers
#and the tips provided by them
import pandas as pd


# In[18]:


#opens and reads the file
open('tips.csv','r').readlines()[:5]


# In[19]:


#reads CSV file and displays first 10 values
tip_file = pd.read_csv('tips.csv')
tip_file.head(10)


# In[20]:


#grouping table to make it more understandable to an enduser
tip_file.groupby('sex').mean()


# In[21]:


#grouping table to make it more understandable to an enduser
tip_file.groupby(['sex','smoker']).mean()


# In[22]:


#now we use pivot table
pd.pivot_table(tip_file,'total_bill','sex','smoker')


# In[23]:


#now we use pivot table and make data even more detailed and presentable
pd.pivot_table(tip_file,'total_bill',['sex','smoker'],['day','time'])


# In[ ]:




