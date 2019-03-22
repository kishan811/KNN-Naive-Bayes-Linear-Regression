#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import csv
import random
import math
import operator
from numpy.linalg import inv
df = pd.read_csv("AdmissionDataset/data.csv")


# In[14]:


df.columns


# In[15]:


from sklearn.model_selection import train_test_split  
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
train,test = train_test_split(df, test_size=0.2) 


# In[16]:


train_y = train['Chance of Admit ']
train = train.drop('Serial No.',axis=1)
train = train.drop('Chance of Admit ',axis=1)
test_y = test['Chance of Admit ']
test = test.drop('Serial No.',axis=1)
test = test.drop('Chance of Admit ',axis=1)
test.head()


# In[17]:


X=train.values
z = np.ones((len(train),1))
X=np.append(z,X, axis=1)


# In[18]:


(X)


# In[19]:


X_trans=X.T


# In[20]:


temp=np.dot(X_trans,X)
inv1=inv(temp)


# In[21]:


# ar=np.array([[1,2,3],[4,5,6]])
# b=np.array([[2],[1],[1]])
# temp2=np.dot(ar,b)
# temp2
# inv1


# In[22]:


temp2=np.dot(inv1,X_trans)


# In[23]:


temp2.shape


# In[26]:


# y.shape


# In[27]:


beta=np.dot(temp2,train_y)


# In[28]:


beta


# In[29]:


# df.head


# In[30]:


X.shape


# In[31]:


z = np.ones((len(test),1))
tt=np.append(z,test, axis=1)
len(tt)


# In[32]:


y_pred=np.dot(tt,beta)


# In[33]:


y_pred.shape


# In[34]:


y_pred


# In[35]:


# test_y


# In[36]:


# np.append(z,X, axis=1)
from sklearn.model_selection import train_test_split  
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 


# # Part 2

# ### Mean squared error

# In[37]:


MSE=((y_pred-test_y)**2).sum()/len(test_y)


# In[38]:


MSE


# ### Mean Absolute Error

# In[39]:


MAE=(abs(y_pred-test_y)).sum()/len(test_y)


# In[40]:


MAE


# ### Mean Percentage Error

# In[41]:


MPE=(((y_pred-test_y)/test_y).sum()/len(test_y))*100


# In[42]:


MPE


# In[43]:


beta


# In[44]:


# import matplotlib.pyplot as plt
# #print plot_arr
# plt.plot(beta)
# plt.show()


# In[ ]:


#     from sklearn.model_selection import train_test_split  
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
#     train,test = train_test_split(df, test_size=0.2) 
# #     random.seed(0)
#     pre_values=[]
#     k = 3
# #     print(X_test)
# #     temp=X_test.to_dict('records')
# #     print (temp[0])
#     test=test.values
#     train=train.values


# In[ ]:





# In[ ]:




