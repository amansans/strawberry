#!/usr/bin/env python
# coding: utf-8

# In[1]:


#in this project we will get weather information from NOAA
#we will conver the file to readable format and play around
import numpy as np
import matplotlib.pyplot as pp
import seaborn


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#importing ftp data over the internet
import urllib.request
urllib.request.urlretrieve('ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt','stations.txt')


# In[6]:


open('stations.txt','r').readlines()[:5]


# In[9]:


#making an empty dictionary
GSN_stations = {}

#get lines with 'GSN' string in them 
#SPLIT those lines 
#JOIN lines with CODE as the key and fields with array loc 4 onwards 
for line in open('stations.txt','r'):
    if 'GSN' in line:
        field = line.split()
        
        GSN_stations[field[0]] = ' '.join(field[4:])


# In[10]:


#find stations
def findstation(s):
    station = {code: name for code,name in stations.items() if s in name}
    print(station)


# In[11]:


findstation('PARIS')


# In[15]:


#we will use the following stations for further procesing
my_stations = ['USW00022536','USW00023188','USW00014922','RSM00030710']


# In[16]:


#this is how the contents look like inside the file
open('USW00022536.dly','r').readlines()[:5]


# In[17]:


#to make the file end-reader friendly
def convert(file_name):
    return np.genfromtxt(file_name,
                         delimiter = file_delimiter,
                         usecols = file_cols,
                         dtype = file_dtype,
                         names = file_names)


# In[18]:


#define length, columns_used, datatypes of all columns and names of each columns
file_delimiter = [11,4,2,4] + [5,1,1,1] * 31
file_cols = [1,2,3] + [4*i for i in range(1,32)]
file_dtype = [np.int32,np.int32,(np.str_,4)] + [np.int32] * 31
file_names = ['year','month','obs'] + [str(day) for day in range(1,31+1)]


# In[20]:


#taking Lihue as an example
lihue = convert('USW00022536.dly')


# In[21]:


lihue


# In[22]:


#defining the date in datetime format
#dates will progress from february 2 onwards with a step of 1 day
#temprature needs to be divided by 10 to get the correct value
def unroll(record):
    startdate = np.datetime64('{}-{:02}'.format(record['year'],record['month']))
    dates = np.arange(startdate,startdate + np.timedelta64(1,'M'),np.timedelta64(1,'D'))
    
    rows = [(date,record[str(i+1)]/10) for i,date in enumerate(dates)]
    
    return np.array(rows,dtype=[('date','M8[D]'),('value','d')])


# In[23]:


unroll(lihue[0])


# In[31]:


#getting minimum and maximum tempratures for all years and concatenating them in a single array
def gettemp(file,value):
    return np.concatenate([unroll(row) for row in convert(file) if row[2] == value])


# In[32]:


lihue_T_Max = gettemp('USW00022536.dly','TMAX')
lihue_T_Min = gettemp('USW00022536.dly','TMIN')


# In[33]:


lihue_T_Max


# In[34]:


lihue_T_Min


# In[36]:


pp.plot(lihue_T_Max['date'],lihue_T_Min['value'])


# In[37]:


#all values with -999.9 are used when data is unknown.
#we will change those values to nan first
def gettemp(file,value):
    data = np.concatenate([unroll(row) for row in convert(file) if row[2] == value])
    
    data['value'][data['value'] == -999.9] = np.nan
    
    return data


# In[38]:


lihue_T_Max = gettemp('USW00022536.dly','TMAX')
lihue_T_Min = gettemp('USW00022536.dly','TMIN')


# In[39]:


pp.plot(lihue_T_Max['date'],lihue_T_Max['value'])
pp.plot(lihue_T_Min['date'],lihue_T_Min['value'])


# In[40]:


#to fill nan values for certain days by using interpolation
def fill_na(data):
    dates_float = data['date'].astype(np.float64)
    
    nan = np.isnan(data['value'])
    
    data['value'][nan] = np.interp(dates_float[nan],dates_float[~nan],data['value'][~nan])


# In[41]:


fill_na(lihue_T_Max)
fill_na(lihue_T_Min)


# In[42]:


pp.plot(lihue_T_Min['date'],lihue_T_Min['value'])


# In[43]:


#to smoothen the values for the plot
def smooth(data,window=10):
    new_val = np.correlate(data['value'],np.ones(window)/window,'same')
    
    pp.plot(data['date'],new_val)


# In[44]:


pp.plot(lihue_T_Min[10000:12000]['date'],lihue_T_Min[10000:12000]['value'])

smooth(lihue_T_Min[10000:12000])
smooth(lihue_T_Min[10000:12000],30)


# In[45]:


#to get sub_plots of all 4 data stations
pp.figure(figsize=(10,6))

for i,code in enumerate(my_stations):
    pp.subplot(2,2,i+1)
    
    smooth(gettemp('{}.dly'.format(code),'TMIN'),365)
    smooth(gettemp('{}.dly'.format(code),'TMAX'),365)
    
    pp.title(GSN_stations[code])
    pp.axis(xmin=np.datetime64('1952'),xmax=np.datetime64('2012'),ymin=-10,ymax=30)

pp.tight_layout()


# In[46]:


#to get data of certain year
def selectyear(data,year):
    start = np.datetime64('{}'.format(year))
    end = start + np.timedelta64(1,'Y')
    
    return data[(data['date'] >= start) & (data['date'] < end)]['value']


# In[47]:


selectyear(lihue_T_Min,1951)


# In[48]:


#to get an array of all minimum tempratures for all years in LIHUE
lihue_all_min = np.vstack([selectyear(lihue_T_Min,year)[:365] for year in range(1951,2014+1)])

#to get an array of all maximum tempratures for all years in LIHUE
lihue_all_max = np.vstack([selectyear(lihue_T_Max,year)[:365] for year in range(1951,2014+1)])


# In[51]:


lihue_all_min[:2]


# In[52]:


#to crosscheck if the code worked for 64 years
lihue_all_min.shape


# In[57]:


lihue_tmin_recordmin = np.min(lihue_all_min,axis=0)
lihue_tmin_recordmax = np.max(lihue_all_min,axis=0)


# In[59]:


#plotting minimum tempratures for all years in LIHUE with a '.'
pp.plot(lihue_tmin_recordmax,'.')


# In[61]:


pp.figure(figsize=(12,4))

#for all days in a year
day_range = np.arange(1,365+1)

#filling the tempratures with a gradient of 0.4
pp.fill_between(day_range,np.min(lihue_all_min,axis=0),np.max(lihue_all_min,axis=0),alpha=0.4)
pp.plot(selectyear(lihue_T_Min,2005))

pp.fill_between(day_range,np.min(lihue_all_max,axis=0),np.max(lihue_all_max,axis=0),alpha=0.4)
pp.plot(selectyear(lihue_T_Max,2005))

pp.axis(xmax=365)


# In[64]:


#repeating for other places
#getting  data for Minneaplois
minneapolis_T_Max = gettemp('USW00014922.dly','TMAX')
minneapolis_T_Min = gettemp('USW00014922.dly','TMIN')


# In[65]:


#repeating for other places
#getting  data for San Diego
sandiego_T_Max = gettemp('USW00023188.dly','TMAX')
sandiego_T_Min = gettemp('USW00023188.dly','TMIN')


# In[66]:


#filling null
fill_na(minneapolis_T_Max)
fill_na(minneapolis_T_Min)
fill_na(sandiego_T_Max)
fill_na(sandiego_T_Min)


# In[67]:


year_range = np.arange(1940,2014+1)


# In[70]:


minneapolis_all_max = np.vstack([selectyear(minneapolis_T_Max,year)[:365] for year in year_range])


# In[71]:


minneapolis_mean = np.mean(minneapolis_all_max,axis=1)


# In[72]:


pp.plot(year_range,minneapolis_mean)


# In[73]:


#warmest year of Minneapolis
minneapolis_warmest = year_range[np.argmax(minneapolis_mean)]
minneapolis_warmest   


# In[74]:


#coldest year of Sann Diego
sandiego_all_min = np.vstack([selectyear(sandiego_T_Min,year)[:365] for year in year_range])
sandiego_mean = np.mean(sandiego_all_min,axis=1)
sandiego_coldest = year_range[np.argmin(sandiego_mean)]
sandiego_coldest


# In[76]:


#comparing warmest year in Minneapolis VS coldest year in San Diego
pp.figure(figsize=(12,4))

days = np.arange(1,366+1)

pp.fill_between(days,
                selectyear(minneapolis_T_Min,minneapolis_warmest),
                selectyear(minneapolis_T_Max,minneapolis_warmest),
                color='g',alpha=0.4)

pp.fill_between(days,
                selectyear(sandiego_T_Min,sandiego_coldest),
                selectyear(sandiego_T_Max,sandiego_coldest),
                color='y',alpha=0.4)

pp.axis(xmax=366)

pp.title('{} in Minneapolis vs. {} in San Diego'.format(minneapolis_warmest,sandiego_coldest))


# In[ ]:




