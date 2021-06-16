#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# In[3]:


Diabetses_Ml=pd.read_csv('C:/Users/user/Desktop/IVY WORK BOOK/PYTHON/Python Datasets/prime _indian diabetes diabetes.csv')


# In[14]:


Diabetses_Ml.head()


# In[5]:


Diabetses_Ml.info()


# In[6]:


Diabetses_Ml.nunique()


# In[103]:


cat_cols=['Pregnancies','Outcome']
con_cols=['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
sel_cols=['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies','Outcome']


# In[13]:


Diabetses_Ml.corrwith(Diabetses_Ml['Glucose'], axis=0)


# In[109]:


Diabetses_Ml.hist('Pregnancies', figsize=(15,6))


# In[108]:


Diabetses_Ml.plot.scatter(x='Outcome', y='Pregnancies', figsize=(15,6))


# In[28]:


Diabetses_Ml.groupby('Pregnancies').size().plot(kind='bar')


# In[90]:


Diabetses_Ml.boxplot(column='DiabetesPedigreeFunction',by='Outcome', figsize=(15,6))


# In[89]:


con1=Diabetses_Ml['Outcome']==0
con2=Diabetses_Ml['DiabetesPedigreeFunction']>1.50
delete=Diabetses_Ml[con1 & con2].index
Diabetses_Ml=Diabetses_Ml.drop(delete)


# In[114]:


Diabetses_Ml.describe(include='all')


# In[98]:


Diabetses_Ml.groupby('Outcome').sum()['Age']


# In[99]:


Diabetses_Ml.isnull().sum()


# In[100]:


Diabetses_Ml.dropna()


# In[116]:


Diabetses_Ml_pickle=pd.to_pickle(Diabetses_Ml,'C:/Users/user/Desktop/IVY WORK BOOK/Diabetses_Ml_pickle.pkl')


# In[106]:


## coorelation with categorical target variable with continuous variable Performing Annova Test
from scipy.stats import f_oneway

for i in con_cols:
    print('*'*60)
    AnnovaCategorical=Diabetses_Ml.groupby('Outcome')[i].apply(list)
    Annovaresult=f_oneway(*AnnovaCategorical)
    print("the p values is", Annovaresult[1])
    if(Annovaresult[1]<0.05):
        print(i,"is correleated with target variable")
    else:
        print(i,"is not correleated with target variable")
                    


# In[ ]:


sel_cols=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies','Outcome']


# In[112]:


## coorelation with categorical target variable with categorical variable Performing Annova Test
from scipy.stats import chi2_contingency

for i in cat_cols:
    crosstabresult=pd.crosstab(index=Diabetses_Ml['Pregnancies'],columns=Diabetses_Ml['Outcome'])
    chiqresult=chi2_contingency(crosstabresult)
    print("The p value is ",chiqresult[1])
    if(chiqresult[1]<0.05):
        print(i," is coreleated with target variable")
    else:
        print(i,"is not coreletaed with target variable")


# In[113]:


Diabetses_Ml.corrwith(Diabetses_Ml['Glucose'], axis=0)


# In[125]:


# Train And Test Split
from sklearn.model_selection import train_test_split
from sklearn import metrics

Target=['Outcome']
Predictors=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
X=Diabetses_Ml[Predictors].values
y=Diabetses_Ml[Target].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=45)


# In[148]:


## Logistic regression

from sklearn.linear_model import LogisticRegression

Target=['Outcome']
Predictors=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
X=Diabetses_Ml[Predictors].values
y=Diabetses_Ml[Target].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35, random_state=870547)
lgf=LogisticRegression(C=3,penalty='l2',solver='newton-cg')
predictmodel=lgf.fit(X_train,y_train)
predictions=predictmodel.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))
print("The Accuracy score ", metrics.f1_score(y_test,predictions, average='micro'))
print("The Alternative Accuracy",metrics.classification_report(y_test,predictions).split()[-2])
      


# In[195]:


## Decision tree Classifier

from sklearn import tree

Target='Outcome'
Predictors=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
X=Diabetses_Ml[Predictors].values
y=Diabetses_Ml[Target].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35, random_state=62)
dcf=tree.DecisionTreeClassifier(max_depth=5,criterion='entropy')
predictmodel=dcf.fit(X_train,y_train)
predictions=predictmodel.predict(X_test)        
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))
print("The Accuracy score ", metrics.f1_score(y_test,predictions, average='micro'))
print("The Alternative Accuracy",metrics.classification_report(y_test,predictions).split()[-2])

Feature_importance=pd.Series(predictmodel.feature_importances_, index=Predictors)
Feature_importance.nlargest(12).plot(kind='barh')


# In[197]:


## plotting decsion trees
from IPython.display import Image
from sklearn import tree
import pydotplus

dot_data=tree.export_graphviz(dcf,out_file=None,feature_names=Predictors,class_names=Target)

Graph=pydotplus.graph_from_dot_data(dot_data)
Image(Graph.create_png())



# In[234]:


## random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

Target='Outcome'
Predictors=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
X=Diabetses_Ml[Predictors].values
y=Diabetses_Ml[Target].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.47, random_state=62)
Rcf=RandomForestClassifier(n_estimators=180,criterion='gini', max_depth=5)
predictmodel=Rcf.fit(X_train,y_train)
predictions=predictmodel.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))
print("The Accuracy score ", metrics.f1_score(y_test,predictions, average='micro'))
print("The Alternative Accuracy",metrics.classification_report(y_test,predictions).split()[-2])

feautre_importance =pd.Series(predictmodel.feature_importances_, index=Predictors)
feautre_importance.nlargest(12).plot(kind='barh')


# In[237]:


## plotting a random foresst classifier

from dtreeplt import dtreeplt

dtree=dtreeplt(model=Rcf.estimators_[60],feature_names=Predictors,target_names=Target)

fig=dtree.view()


# In[292]:


## Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

Target='Outcome'
Predictors=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
X=Diabetses_Ml[Predictors].values
y=Diabetses_Ml[Target].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30, random_state=859)

dcf=DecisionTreeClassifier(max_depth=3)
Ada=AdaBoostClassifier(n_estimators=150,learning_rate=0.02,base_estimator=dcf)
predictmodel=Ada.fit(X_train,y_train)
predictions=predictmodel.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))
print("The Accuracy score ", metrics.f1_score(y_test,predictions, average='micro'))
print("The Alternative Accuracy",metrics.classification_report(y_test,predictions).split()[-2])

feautre_importance=pd.Series(predictmodel.feature_importances_,index=Predictors)
feautre_importance.nlargest(12).plot(kind='barh')
    



# In[295]:


## plotting a Ada Boost tree

from dtreeplt import dtreeplt
dtree=dtreeplt(model=Ada.estimators_[30], feature_names=Predictors,target_names=Target)
fig=dtree.view()


# In[317]:


##Xgboost Algorithm

from xgboost import XGBClassifier

Target='Outcome'
Predictors=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
X=Diabetses_Ml[Predictors].values
y=Diabetses_Ml[Target].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.41, random_state=9578)

xgb=XGBClassifier(max_depth=3, learning_rate=0.02, n_estimators=200, objective='binary:logistic', booster='gbtree')
predictmodel=xgb.fit(X_train,y_train)
predictions=predictmodel.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))
print("The Accuracy score",metrics.f1_score(y_test,predictions,average='micro'))
print("The Alternative Accuracy",metrics.classification_report(y_test,predictions).split()[-2])

Feautre_importance=pd.Series(predictmodel.feature_importances_,index=Predictors)
Feautre_importance.nlargest(12).plot(kind='barh')


# In[322]:


## plooting tree 
from xgboost import plot_tree
import matplotlib.pyplot as plt

fig, ax= plt.subplots(figsize=(20,8))
plot_tree(xgb, num_trees=150, ax=ax)




# In[356]:


## KNN classifier

from sklearn.neighbors import KNeighborsClassifier

Target='Outcome'
Predictors=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
X=Diabetses_Ml[Predictors].values
y=Diabetses_Ml[Target].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.34, random_state=8457)


knn=KNeighborsClassifier(n_neighbors=5)
predictmodel=knn.fit(X_train,y_train)
predictions=predictmodel.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))
print("The Accuracy score",metrics.f1_score(y_test,predictions,average='micro'))
print("The Alternative Accuracy",metrics.classification_report(y_test,predictions).split()[-2])


# In[397]:


## SVM
from sklearn import svm

Target='Outcome'
Predictors=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
X=Diabetses_Ml[Predictors].values
y=Diabetses_Ml[Target].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.45, random_state=500000)


svm=svm.SVC(C=20,kernel='rbf',gamma=0.01)
predictmodel=svm.fit(X_train,y_train)
predictions=predictmodel.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))
print("The Accuracy score",metrics.f1_score(y_test,predictions,average='micro'))
print("The Alternative Accuracy",metrics.classification_report(y_test,predictions).split()[-2])



# In[409]:


## Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB

Target='Outcome'
Predictors=['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Pregnancies']
X=Diabetses_Ml[Predictors].values
y=Diabetses_Ml[Target].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.40,random_state=218000)


NB=GaussianNB()
predictmodel=NB.fit(X_train,y_train)
predictions=predictmodel.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))

print("The Accuracy score",metrics.f1_score(y_test,predictions,average='micro'))
print("The Alternative Accuracy",metrics.classification_report(y_test,predictions).split()[-2])


# In[ ]:




