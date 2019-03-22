#!/usr/bin/env python
# coding: utf-8

# In[621]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import csv
import random
import math
import operator
import sys


# In[622]:


df = pd.read_csv('Iris/Iris.csv') 


# In[623]:


df.columns=['a','b','c','d','label']
df.head()


# In[624]:


X = df.iloc[:, :-1].values  
y = df.iloc[:, 4].values  


# In[625]:


print(np.unique(y))


# In[626]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 


# In[627]:


# X_train.head()


# In[628]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=25)  
classifier.fit(X_train, y_train) 


# In[629]:


y_pred = classifier.predict(X_test)


# In[630]:


print(y_pred)


# In[631]:


# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))  
# print(classification_report(y_test, y_pred)) 
# print(accuracy_score(y_test, y_pred))  


# In[632]:


def euclidDistance(val1, val2, entirelength):
    distance = 0
    for x in range(entirelength):
        distance += pow((val1[x] - val2[x]), 2)
    return math.sqrt(distance)


# In[633]:


def manhattanDistance(val1, val2, entirelength):
    distance = 0
    for x in range(entirelength):
        distance += abs(val1[x] - val2[x])
    return distance


# In[634]:


def chebyshevDistance(val1, val2, entirelength):
    distance = 0
    for x in range(entirelength):
        distance = max(distance,abs(val1[x] - val2[x]))
    return distance


# In[635]:


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


# In[636]:


def resultss(neighbors):
    predlabel = {}
    for x in range(len(neighbors)):
        res = neighbors[x][-1]
        if res not in predlabel:
            predlabel[res] = 1
        predlabel[res] += 1
    ans = sorted(predlabel.items(), reverse=True)
    return ans[0][0]


# In[651]:


def accu_calc(test, predlabel):
    correct = 0
    for x in range(len(test)):
        if test[x][-1] == predlabel[x]:
            correct += 1
    return (correct/float(len(test))) * 100.0


# ### Euclidean calculations

# In[638]:


X = df.iloc[:, :-1].astype(float)  
#     print(X)
y = df.iloc[:, 4].values
#     print(y)
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
train,test = train_test_split(df, test_size=0.2) 
#     random.seed(0)
pre_values=[]
k = 5
#     print(X_test)
#     temp=X_test.to_dict('records')
#     print (temp[0])
fn=sys.argv[1]
df2=pd.read_csv(fn)
df2.columns=['a','b','c','d','label']
test=df2
test1=test
train1=train
test=test.values
train=train.values
#     test_y=test('label')
#     print(len(train))
#     print(test)
#     for x in range(len(X_test)):
#         testSet.append()
for x in range(len(test)):
    neighbors = Neighbor_points(train, test[x], k)
#         print(neighbors[x][-1])
    result = resultss(neighbors)
    pre_values.append(result)
#         print('> predicted value by algo=' + result + ' and actual value by scikit=' + test[x][-1])
accuracy = accu_calc(test, pre_values)
print ('The Accuracy is: ', accuracy)


# In[639]:


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
    acc.append(accuracy)
# print(acc)
print ('The Accuracy are: ', acc)
#     print(knn)


# In[640]:


# plt.figure(figsize=(12, 6))  
plt.plot(range(1, 25), acc)
plt.title('Acc vs K Value')  
plt.xlabel('K Value')
plt.grid(True)
plt.ylabel('Accuracy')  


# ### Manhattan calculations

# In[641]:


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


# In[642]:


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
accuracy = accu_calc(test, pre_values)
print ('The Accuracy is: ', accuracy)


# In[643]:


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
print ('The Accuracy are: ', acc)
#     print ('The Accuracy is: ', accuracy)
#     print(knn)


# In[644]:


# plt.figure(figsize=(12, 6))  
plt.plot(range(1, 25), acc)
plt.title('Acc vs K Value')  
plt.xlabel('K Value')
plt.grid(True)
plt.ylabel('Accuracy')  


# ### Chebyshev calculations

# In[645]:


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


# In[646]:


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


# In[647]:


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

print ('The Accuracy are: ', acc)
# print(acc)
#     print ('The Accuracy is: ', accuracy)
#     print(knn)

# plt.figure(figsize=(12, 6))  
plt.plot(range(1, 25), acc)
plt.title('Acc vs K Value')  
plt.xlabel('K Value')
plt.grid(True)
plt.ylabel('Accuracy')  


# In[648]:


train1.head()


# In[649]:


y_train=train1.pop('label')
X_train=train1
y_test=test1.pop('label')
X_test=test1


# ### Accuracy using SKlearn

# In[650]:


# y_train=X_train.pop('label')
# print(X_train.head())
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred))  


# In[ ]:




