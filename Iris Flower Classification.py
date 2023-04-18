#!/usr/bin/env python
# coding: utf-8

# ## Iris Flower Classification

# ### Import Modules

# In[1]:


import warnings
warnings.filterwarnings('ignore')  ## To avoid warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Loading Dataset

# In[7]:


df = pd.read_csv("Iris.csv")
df.head()


# In[8]:


## To Drop Id column
df.drop(columns = ['Id'], inplace = True)


# In[9]:


df.head()


# In[11]:


## To know the shape of a dataset
df.shape


# In[13]:


## To know the information about the dataset
df.info()


# In[15]:


## To count no. of samples 
df['Species'].value_counts()


# ### Preprocessing the data set

# In[17]:


## Checking null values
df.isnull().sum()


# ### Exploratory Data Analysis 

# #### Scatter Plot

# In[22]:


sns.FacetGrid(df, hue = 'Species', height = 6).map(plt.scatter, "PetalLengthCm","SepalWidthCm").add_legend()


# In[24]:


sns.FacetGrid(df, hue = 'Species', height = 6).map(plt.scatter, "PetalLengthCm","PetalWidthCm").add_legend()


# In[28]:


sns.FacetGrid(df, hue = 'Species', height = 6).map(plt.scatter, "SepalLengthCm","SepalWidthCm").add_legend()


# ### Correlation Matrix

# In[29]:


df.corr()


# #### Heatmap

# In[30]:


corr = df.corr()
fig, ax = plt.subplots(figsize = (5,4))
sns.heatmap(corr,annot = True, ax = ax)


# ### Data Preprocessing

# #### Label Encoder

# In[33]:


from sklearn.preprocessing import LabelEncoder
L = LabelEncoder()


# In[34]:


df['Species'] = L.fit_transform(df['Species'])


# In[36]:


df.head()


# ### Model Training

# In[97]:


from sklearn.model_selection import train_test_split

x = df.iloc[:,0:4]
y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# ### Logistic Regression

# In[98]:


from sklearn.linear_model import LogisticRegression


# In[99]:


model = LogisticRegression()


# In[100]:


## Model training
model.fit(x_train, y_train)


# In[101]:


## To get performance
print("Accuracy : ", model.score(x_test, y_test)*100)


# #### Predictions

# In[102]:


expected = y
predicted = model.predict(x)
predicted


# In[103]:


from sklearn import metrics 


# In[104]:


print(metrics.classification_report(expected,predicted))


# In[105]:


print(metrics.confusion_matrix(expected,predicted))


# ### KNN - K Nearest Neighbourhood

# In[106]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[107]:


model.fit(x_train, y_train)


# In[108]:


print("Accuracy : ", model.score(x_test, y_test)*100)


# ### Naive Bayes Classifier

# In[109]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[110]:


model.fit(x_train, y_train)


# In[111]:


print("Accuracy : ", model.score(x_test, y_test)*100)


# In[ ]:




