#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df_train=pd.read_csv("Training.csv")
df_test=pd.read_csv("Testing.csv")


# In[3]:


df_train.head()


# In[4]:


df_test.head()


# In[5]:


df_train=df_train.drop('Unnamed: 133',axis=1)


# In[7]:


df_train.head()


# In[9]:


df_train.shape


# In[10]:


df_test.shape


# In[15]:


df_train.isnull().sum()


# In[18]:


df_train.info()


# In[19]:


df_train.describe()


# In[20]:


df_train.columns


# In[21]:


import seaborn as sns


# In[22]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[23]:


x=df_train.drop('prognosis',axis=1)


# In[24]:


x.head()


# In[26]:


y=df_train['prognosis']


# In[29]:


y.describe()


# In[32]:


X=df_test.drop('prognosis',axis=1)
X.head()


# In[33]:


Y=df_test['prognosis']


# In[34]:


Y.head()


# In[35]:


np.std(x,axis=0)


# In[36]:


np.std(x,axis=1)


# In[37]:


correlation_matrix=x.corr()
correlation_matrix


# In[38]:


prognosis_counts=df_train['prognosis'].value_counts()
prognosis_counts


# In[39]:


label=prognosis_counts.index
value_count=prognosis_counts.values


# In[40]:


import matplotlib.pyplot as plt


# In[44]:


fig=plt.figure(figsize=(30,28))
plt.pie(value_count,labels=label,autopct='%1f%%')
plt.title('Prognosis')
plt.show()


# In[46]:


plt.figure(figsize=(30,28))
sns.heatmap(x.corr(),annot=False,linewidth=1,cmap='coolwarm')


# In[47]:


for column in df_train.columns:
    fig=plt.figure(figsize=(5,4))
    sns.countplot(data=df_train,x=column)
    plt.xticks(rotation=90)
    plt.xlabel(column)
    plt.show()


# In[49]:


sns.histplot(x='itching', data=df_train,kde=True) 
#kde-Kernel Density Estimate


# In[50]:


sns.barplot(x=df_train["skin_rash"] , y=df_train["itching"] , data=df_train,hue="prognosis")


# In[51]:


sns.barplot(x=df_train["small_dents_in_nails"] , y=df_train["inflammatory_nails"] , data=df_train)


# In[54]:


sns.boxplot(x=df_train["red_sore_around_nose"],y=df_train["continuous_sneezing"],data=df_train)


# In[55]:


df_train.head()


# In[56]:


y=encoder.fit_transform(y)
Y=encoder.fit_transform(Y)


# In[58]:


y


# In[59]:


Y


# In[60]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[63]:


rf=RandomForestClassifier(n_estimators=300,random_state=42,max_depth=25)
rf.fit(x_train,y_train)


# In[64]:


y_pred=rf.predict(x_test)


# In[65]:


y_pred


# In[67]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[68]:


Y


# In[69]:


X


# In[76]:


test_prediction=rf.predict(X)


# In[77]:


test_prediction


# In[78]:


print(accuracy_score(Y,test_prediction))


# In[79]:


type(Y)


# In[80]:


Y


# In[81]:


Y_df=pd.DataFrame(Y,columns=["prognosis"])
test_pred_df=pd.DataFrame(test_prediction,columns=["predicted"])


# In[82]:


result_df=Y_df.join(test_pred_df)
result_df


# In[83]:


from sklearn.tree import DecisionTreeClassifier


# In[87]:


dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)


# In[90]:


dtree_pred=dtree.predict(x_test)


# In[91]:


print(accuracy_score(y_test,dtree_pred))


# In[92]:


dtree.pred.shape


# In[93]:


Y.shape


# In[94]:


dtree_pred_test=dtree.predict(X)
print(accuracy_score(Y,dtree_pred_test))


# In[95]:


#Scaling using svm


# In[96]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[97]:


scaled_x_train=scaler.fit_transform(x_train)
scaled_x_test=scaler.transform(x_test)
scaled_x=scaler.transform(X)


# In[98]:


from sklearn.svm import SVC


# In[100]:


svm_clf=SVC(kernel='linear',random_state=52,C=100)
svm_clf=svm_clf.fit(scaled_x_train,y_train)
svm_clf


# In[101]:


pred_y=svm_clf.predict(scaled_x_test)
print(accuracy_score(y_test,pred_y))


# In[102]:


pred_x_test=svm_clf.predict(scaled_x)


# In[103]:


print(accuracy_score(Y,pred_x_test))


# In[ ]:





# In[ ]:




