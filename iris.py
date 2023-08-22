#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeClassifier


# In[15]:


iris = datasets.load_iris()


# In[16]:


print(iris)


# In[17]:


data = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[18]:


# Data Preprocessing is not considered


# In[19]:


label = iris.target
print(label)


# In[20]:


# Train phase


# In[21]:


dtree = DecisionTreeClassifier()
dtree.fit(data, label)


# In[22]:


# Predict sample data


# In[23]:


dtree.predict([
    [1.1, 1.8, 1.2, 0.2],
    [2.1, 5.1, 0.5, 2.9]
])


# In[ ]:




