#!/usr/bin/env python
# coding: utf-8

# In[11]:


#we will work on a file containing the data of popularity of names in US 
#from the years 1880 to 2014
import numpy as np
import matplotlib.pyplot as pp
import pandas as pd
import seaborn


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


#extract all the contents of the zipfile in the same directory
import zipfile
zipfile.ZipFile('names.zip').extractall('.')


# In[14]:


#to get names of all the files in 'names'
import os 
os.listdir('names')


# In[15]:


#open file in read mode
open('names/yob2011.txt','r').readlines()[:5]


# In[17]:


#converting CSV file to panda DataFrame
#giving column names with 'names' attribute
namelist_2011 = pd.read_csv('names/yob2011.txt',names=['name','sex','number'])
namelist_2011.head()


# In[18]:


#to concatenate the dataframes from the year 1880 to 2014 into single dataframe
allyears_name = []

for year in range(1880,2014+1):
    allyears_name.append(pd.read_csv('names/yob{}.txt'.format(year),names=['name','sex','number']))
    allyears_name[-1]['year'] = year

allyears = pd.concat(allyears_name)


# In[20]:


allyears.tail()


# In[22]:


#setting multiple indexes
allyears_indexed = allyears.set_index(['sex','name','year']).sort_index()
allyears_indexed


# In[24]:


#function to find someone usinf 'sex' and 'name' as input and plotting them using matplotlib.plot
def name_plot(sex,name):
    data = allyears_indexed.loc[sex,name]
    
    pp.plot(data.index,data.values)


# In[32]:


pp.figure(figsize=(12,2.5))

name_list = ['Chiara','Claire','Clare','Clara','Ciara']

for name in name_list:
    name_plot('F',name)

pp.legend(name_list)


# In[45]:


pp.figure(figsize=(12,2.5))

name_plot = ['Michael','John','David','Martin']

for name in name_plot:
    plotname('M',name)

pp.legend(name_plot)


# In[ ]:


|

