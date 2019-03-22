#!/usr/bin/env python
# coding: utf-8

# In[282]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import csv
import random
import math
import operator
import sys


# In[283]:


df=pd.read_csv('Robot1',header=None,delimiter=r'\s+')


# In[284]:


temp=df[0]
df=df.drop([0,7],axis=1)
df=df.join(temp)
# df.loc[:,0]


# In[285]:


# df


# In[286]:


def euclidDistance(val1, val2, entirelength):
    distance = 0
    for x in range(entirelength):
        distance += pow((val1[x] - val2[x]), 2)
    return math.sqrt(distance)


# In[287]:


def chebyshevDistance(val1, val2, entirelength):
    distance = 0
    for x in range(entirelength):
        distance += max(distance,abs(val1[x] - val2[x]))
    return distance


# In[288]:


def manhattanDistance(val1, val2, entirelength):
    distance = 0
    for x in range(entirelength):
        distance += abs(val1[x] - val2[x])
    return distance


# In[289]:


def Neighbor_points(train_d, test_d, k):
    distances = []
    length = len(test_d)-1
    for x in range(len(train_d)):
        dist = euclidDistance(test_d, train_d[x], length)
        distances.append((train_d[x], dist))
    distances.sort(key=lambda x: x[1])
#     print(distances)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
#         print(neighbors)
    return neighbors


# In[290]:


def Neighbor_points2(train_d, test_d, k):
    distances = []
    length = len(test_d)-1
    for x in range(len(train_d)):
        dist = chebyshevDistance(test_d, train_d[x], length)
        distances.append((train_d[x], dist))
    distances.sort(key=lambda x: x[1])
#     print(distances)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
#         print(neighbors)
    return neighbors


# In[291]:


def Neighbor_points1(train_d, test_d, k):
    distances = []
    length = len(test_d)-1
    for x in range(len(train_d)):
        dist = manhattanDistance(test_d, train_d[x], length)
        distances.append((train_d[x], dist))
    distances.sort(key=lambda x: x[1])
#     print(distances)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
#         print(neighbors)
    return neighbors


# In[292]:


def resultss(neighbors):
    predlabel = {}
#     print(neighbors)
    for x in range(len(neighbors)):
        res = neighbors[x][-1]
        if res not in predlabel:
            predlabel[res] = 1
        predlabel[res] += 1
    ans = sorted(predlabel.items(), reverse=True)
    return ans[0][0]


# In[293]:


def accu_calc(test, predlabel):
    correct = 0
    tp,fp,tn,fn=0,0,0,0
    for x in range(len(test)):
        if test[x][-1] == predlabel[x]:
            correct += 1
        if test[x][-1]==1 and predlabel[x]==1:
            tp+=1
        if test[x][-1]==0 and predlabel[x]==1:
            fp+=1
        if test[x][-1]==0 and predlabel[x]==0:
            tn+=1
        if test[x][-1]==1 and predlabel[x]==0:
            fn+=1
    x = (tp+fn)
    y = (tp+fp)
    if x:
        rc=tp/x
    if y:
        pc=tp/y
    f1=(2*rc*pc)/(rc+pc)
    print("\nRecall: ", rc)
    print("\nPrecision: ", pc)
    print("\nF1-Score: ", f1)

    print("\nTrue pos: ",tp)
    print("\nFalse pos: ",fp)
    print("\nTrue neg: ",tn)
    print("\nFalse neg: ",fn)
    accuracy=(correct/float(len(test))) * 100.0
    return accuracy

#         print(neighbors)
#     return neighbors


# ### Euclidean Calculations

# In[294]:


from sklearn.model_selection import train_test_split  
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
train,test = train_test_split(df, test_size=0.2) 
#     random.seed(0)
pre_values=[]
k = 5
#     print(X_test)
#     temp=X_test.to_dict('records')
#     print (temp[0])
fn=sys.argv[1]
df2=pd.read_csv(fn)
# df2.columns=['a','b','c','d','label']
test=df2

test1=test
train1=train
test=test.values
train=train.values
#     print(len(train))
#     print(test)
#     for x in range(len(X_test)):
#         testSet.append()
#     neighbors=[]
for x in range(len(test)):
    neighbors = Neighbor_points(train, test[x], k)
#         print(neighbors[x][-1])
    result = resultss(neighbors)
#         print(result)
    pre_values.append(result)
#         print('> predicted value by algo=' + result + ' and actual value by scikit=' + test[x][-1])
accuracy = accu_calc(test, pre_values)
print ('The Accuracy is: ', accuracy)


# In[295]:


acc=[]
for i in range(1, 25):
    pre_values=[]
    knn = []
    result=[]
    for x in range(len(test)):
        knn = Neighbor_points(train, test[x], i)
#         print(knn)
        result = resultss(knn)
        pre_values.append(result)
#         print('> predicted value by algo=' + result + ' and actual value by scikit=' + test[x][-1])
    accuracy = accu_calc(test, pre_values)
    acc.append(accuracy)# plt.figure(figsize=(12, 6))  
    


# In[296]:


print ('The Euclid Accuracy are: ', acc)

plt.plot(range(1, 25), acc)
plt.title('Acc vs K Value')  
plt.xlabel('K Value')
plt.grid(True)
plt.ylabel('Accuracy')  


# ### Manhattan Calculations

# In[297]:


X = df.iloc[:, :-1].astype(float)  
#     print(X)
y = df.iloc[:, 4].values
#     print(y)
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
#     train,test = train_test_split(df, test_size=0.2) 
#     random.seed(0)
pre_values=[]
k = 5
#     print(X_test)
#     temp=X_test.to_dict('records')
#     print (temp[0])
#     test1=test
#     train1=train
#     test=test.values
#     train=train.values
#     test_y=test('label')
#     print(len(train))
#     print(test)
#     for x in range(len(X_test)):
#         testSet.append()
for x in range(len(test)):
    neighbors = Neighbor_points1(train, test[x], k)
#         print(neighbors[x][-1])
    result = resultss(neighbors)
    pre_values.append(result)
#         print('> predicted value by algo=' + result + ' and actual value by scikit=' + test[x][-1])
#     accuracy = accu_calc(test, pre_values)
print ('The Accuracy is: ', accuracy)


# In[298]:


acc=[]
for i in range(1, 25):
    pre_values=[]
    knn = []
    result=[]
    for x in range(len(test)):
        knn = Neighbor_points1(train, test[x], i)
#         print(knn)
        result = resultss(knn)
        pre_values.append(result)
#         print('> predicted value by algo=' + result + ' and actual value by scikit=' + test[x][-1])
    accuracy = accu_calc(test, pre_values)
    acc.append(accuracy)
# print(acc)


# In[299]:


print ('The Manhattan Accuracy are: ', acc)
#     print ('The Accuracy is: ', accuracy)
#     print(knn)
# plt.figure(figsize=(12, 6))  
plt.plot(range(1, 25), acc)
plt.title('Acc vs K Value')  
plt.xlabel('K Value')
plt.grid(True)
plt.ylabel('Accuracy')  


# ### Chebyshev Calculations

# In[300]:


X = df.iloc[:, :-1].astype(float)  
#     print(X)
y = df.iloc[:, 4].values
#     print(y)
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
#     train,test = train_test_split(df, test_size=0.2) 
#     random.seed(0)
pre_values=[]
k = 5
#     print(X_test)
#     temp=X_test.to_dict('records')
#     print (temp[0])
#     test1=test
#     train1=train
#     test=test.values
#     train=train.values
#     test_y=test('label')
#     print(len(train))
#     print(test)
#     for x in range(len(X_test)):
#         testSet.append()
for x in range(len(test)):
    neighbors = Neighbor_points2(train, test[x], k)
#         print(neighbors[x][-1])
    result = resultss(neighbors)
    pre_values.append(result)
#         print('> predicted value by algo=' + result + ' and actual value by scikit=' + test[x][-1])
accuracy = accu_calc(test, pre_values)
print ('The Accuracy is: ', accuracy)


# In[301]:


acc=[]
for i in range(1, 25):
    pre_values=[]
    knn = []
    result=[]
    for x in range(len(test)):
        knn = Neighbor_points2(train, test[x], i)
#         print(knn)
        result = resultss(knn)
        pre_values.append(result)
#         print('> predicted value by algo=' + result + ' and actual value by scikit=' + test[x][-1])
    accuracy = accu_calc(test, pre_values)
    acc.append(accuracy)



# In[302]:


print ('The Chebyshev Accuracy are: ', acc)
# print(acc)
#     print ('The Accuracy is: ', accuracy)
#     print(knn)

# plt.figure(figsize=(12, 6))  
plt.plot(range(1, 25), acc)
plt.title('Acc vs K Value')  
plt.xlabel('K Value')
plt.grid(True)
plt.ylabel('Accuracy')  


# In[303]:


y_train=train1.pop(0)
X_train=train1
y_test=test1.pop(0)
X_test=test1
train1.head()


# ### Accuracy using SKlearn

# In[305]:


X = df.iloc[:, :-1].values 
# print(X)
y = df.iloc[:, 6].values  
# print(y)
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred))  


# In[ ]:




