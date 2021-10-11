#!/usr/bin/env python
# coding: utf-8

# In[1]:


#in this file we will create panels
#we will use Panels to represent data of Apple, Microsft, Dell and Google stocks 
#we import necessary libraries
#we set printing precision to 2 decimal places
import pandas as pd
import numpy as np
import datetime
from pandas_datareader import data, wb

pd.set_eng_float_format(accuracy = 2, use_eng_prefix =True)


# In[11]:


#making a panel from Panda.Panel
my_panel = pd.Panel(np.random.randn(2,5,4),
                   items = ['primary','secondary'],
                   major_axis=pd.date_range('20180201', periods= 5),
                   minor_axis = ['A','B','C','D'])
my_panel


# In[10]:


#making a Panel from dictionary consisting of DataFrames
dict_DF = {'dict1' : pd.DataFrame(np.random.randn(4,3)),
           'dict2' : pd.DataFrame(np.random.randn(4,2))}
           
P_dict_DF = pd.Panel(dict_DF)
P_dict_DF


# In[24]:


#getting stock data from yahoo as a Panel for year 2010 to 2019
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2019, 1, 1)
Panel_data = pd.Panel(dict((stk,data.DataReader("F",'yahoo',start_date,end_date))
                           for stk in ['AAPL', 'GOOG', 'MSFT', 'DELL']))

Panel_data


# In[40]:


Panel_data = Panel_data.swapaxes('items', 'minor')
Panel_data['Adj Close'].head(10)


# In[36]:


#To represent all rows and columns for 12th AUG 2016
Panel_data.ix[:, '7/12/2016', :]


# In[42]:


#multikey formatting from Panel to Frame
stacked_data = Panel_data.ix[:, '6/30/2016':, :].to_frame()
stacked_data


# In[43]:


#changing data back from stacked to panel
stacked_data.to_panel()


# In[ ]:




