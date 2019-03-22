#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import csv
import random
import math
import sys
import operator
df = pd.read_csv("LoanDataset/data.csv", skipinitialspace=True)


# In[122]:


df.columns=['id','age','expyears','annuinc','zip','famsize','avgspend','edulevel','mortval','label','security','COD','netbank','CC']
features=df.columns
print(features)


# In[123]:


df=df.drop('id', 1)
df=df.drop('zip', 1)


# In[124]:


from sklearn.model_selection import train_test_split  
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
train,test = train_test_split(df, test_size=0.2) 
fn=sys.argv[1]
df2=pd.read_csv(fn)
df2.columns=['id','age','expyears','annuinc','zip','famsize','avgspend','edulevel','mortval','label','security','COD','netbank','CC']
test=df2
# df


# In[125]:


len(train)


# In[126]:


import math
def continuous_prob(x, mean, stdev):
    exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exp


# In[127]:


import math
def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stddev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


# In[128]:


# test


# In[129]:


numerical_attr = ['age','expyears','annuinc','avgspend','edulevel','mortval']
categorical_attr = ['famsize','security','COD','netbank','CC']


# In[130]:


mean={}
std_dev={}
for i in numerical_attr:
    mean[i]={}
    std_dev[i]={}
    yess=train[train['label']==1][i].mean()
    mean[i]['yes']=yess;
    noss=train[train['label']==0][i].mean()
    mean[i]['no']=noss
    posdev=train[train['label']==1][i].std()
    std_dev[i]['yes']=posdev
    negdev=train[train['label']==0][i].std()
    std_dev[i]['no']=negdev;


# In[131]:


categorical={}
for i in categorical_attr:
    categorical[i]={}
    unique_feature=train[i].unique()
    for x in unique_feature:
        categorical[i][x]={}
#         print(train[train[i]==x])
        data=train[train[i]==x]
#         print(data)
        p=len(train[train['label']==1][i])
        q=len(train[train['label']==0][i])
        yess=data[data['label']==1][i].count()/p
        #print(yes_count)
        categorical[i][x]['yes']=yess
        noss=data[data['label']==0][i].count()/q
        #print(no_count)
        categorical[i][x]['no']=noss
        
    #print("new")   
totyes = len(train[train['label']==1])
pyess=totyes/len(train);
pnoo=1-pyess    


# In[ ]:





# In[132]:


pred=[]
for index,data in test.iterrows():
    pp=1
    pn=1
    for cat in categorical_attr:
#         print(cat)
#         print(data[cat])
        pp=pp*categorical[cat][data[cat]]['yes']
        pp=pn*categorical[cat][data[cat]]['no']
    for num in numerical_attr:
        mean_y=mean[num]['yes']
        std_y=std_dev[num]['yes']
        p_pos=continuous_prob(data[num],mean_y,std_y)
        pp=pp*p_pos
        mean_n=mean[num]['no']
        std_n=std_dev[num]['no']
        n_nos=continuous_prob(data[num],mean_n,std_n);
        pn=pn*n_nos
    pp =pp*pyess;
    pn=pn*pnoo
    if pp>pn:
        pred.append(1)
    else:
        pred.append(0)


# In[133]:


testlist = test['label'].tolist()
len(testlist)


# In[134]:


count=0
for i in range(len(testlist)):
    if(testlist[i]==pred[i]):
        count+=1
print (count/len(testlist)) 


# In[135]:


df.head()


# In[136]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# df=df.drop('id', 1)
# df=df.drop('zip', 1)
trax = train[['age','expyears','annuinc','famsize','avgspend','edulevel','mortval','security','COD','netbank','CC']]
tray = train[['label']]
# print(tray)
trax = np.array(trax)
tray = np.array(tray)

clf = GaussianNB()
clf.fit(trax,tray )

testx = test[['age','expyears','annuinc','famsize','avgspend','edulevel','mortval','security','COD','netbank','CC']]
testy = test[['label']]
# testx = validate.iloc[:,:-1]
# testy = validate.iloc[:,-1]

testx = np.array(testx)
testy = np.array(testy)

predicts = clf.predict(testx)

# y = np.array(training_labels)

print ("Accuracy Rate is: %f" % accuracy_score(testy, predicts))
print(confusion_matrix(testy, predicts))  
print(classification_report(testy, predicts)) 


# In[ ]:





# In[ ]:





# In[ ]:




