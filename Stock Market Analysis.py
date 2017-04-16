
# coding: utf-8

# In[1]:

import pandas as pd
from pandas import Series,DataFrame
import numpy as np


# In[2]:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[3]:

get_ipython().magic('matplotlib inline')


# In[4]:

from pandas_datareader import data, wb


# In[5]:

from datetime import datetime


# In[6]:

tech_list=['AAPL','GOOG','MSFT','AMZN']


# In[7]:

end=datetime.now()


# In[8]:

start=datetime(end.year-1,end.month,end.day)


# In[9]:

start


# In[10]:

end


# In[11]:

AAPL=data.DataReader('AAPL','yahoo',start,end)
GOOG=data.DataReader('GOOG','yahoo',start,end)
MSFT=data.DataReader('MSFT','yahoo',start,end)
AMZN=data.DataReader('AMZN','yahoo',start,end)


# In[12]:

AAPL


# In[13]:

AAPL.describe()


# In[14]:

#用adj close分析历史数据


# In[15]:

AAPL.info()


# In[16]:

AAPL['Adj Close'].plot(legend=True,figsize=(10,4))


# In[17]:

AAPL['Volume'].plot(legend=True,figsize=(10,4))


# In[18]:

ma_day=[10,20,50]
for ma in ma_day:
    column_name='MA for %s days' %(str(ma))
    AAPL[column_name]=pd.rolling_mean(AAPL['Adj Close'],ma)


# In[ ]:




# In[19]:

#Daily return
AAPL['Daily Return']=AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(10,4),legend=True,linestyle='--',marker='o')


# In[20]:

sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# In[21]:

AAPL['Daily Return'].hist(bins=100)


# In[22]:

closing_df=data.DataReader(tech_list,'yahoo',start,end)['Adj Close']


# In[23]:

closing_df.head()


# In[24]:

tech_rets=closing_df.pct_change()


# In[25]:

tech_rets.head()


# In[26]:

sns.jointplot('GOOG','GOOG',tech_rets,kind='scatter',color='seagreen')


# In[27]:

sns.jointplot('GOOG','MSFT',tech_rets,kind='scatter')


# In[28]:

tech_rets.head()


# In[29]:

sns.pairplot(tech_rets.dropna())


# In[30]:

returns_fig=sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)


# In[31]:

returns_fig=sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)


# In[32]:

sns.corrplot(tech_rets.dropna(),annot=True)


# In[ ]:

#risk analysis


# In[36]:

# Let's start by defining a new DataFrame as a clenaed version of the oriignal tech_rets DataFrame
rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(),alpha = 0.5,s =area)

# Set the x and y limits of the plot (optional, remove this if you don't see anything in your plot)
plt.ylim([0.01,0.025])
plt.xlim([-0.003,0.004])

#Set the plot axis titles
plt.xlabel('Expected returns')
plt.ylabel('Risk')

# Label the scatter plots, for more info on how this is done, chekc out the link below
# http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


# In[39]:

import plotly
import cufflinks as cf
cf.go_offline()


# In[45]:

tech_stocks = pd.concat([AAPL, GOOG, MSFT, AMZN],axis=1,keys=tech_list)


# In[46]:

tech_stocks.columns.names = ['Tech Ticker','Stock Info']


# In[47]:

tech_stocks


# In[49]:

tech_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()


# In[67]:

close_corr = tech_stocks.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdbu')


# In[70]:

AAPL[['Open', 'High', 'Low', 'Close']].ix[start:end].iplot(kind='candle')


# In[72]:

GOOG['Close'].ix[start:end].ta_plot(study='sma',periods=[15,50,60],title='Simple Moving Averages')


# In[73]:

MSFT['Close'].ix[start:end].ta_plot(study='boll')


# In[ ]:



