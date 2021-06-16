#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_rows', 200000)
pd.set_option('display.max_columns', 200)
np.set_printoptions(suppress=True)


# In[3]:


dftrain=pd.read_csv('Train.csv')
dftest=pd.read_csv('Test.csv')
target=pd.read_csv('Target.csv')


# In[5]:


dftrain.shape


# In[46]:


target.tail(5)


# In[ ]:





# In[7]:


dftest.shape


# In[8]:


dftrain.shape


# In[9]:


dftest.isnull().sum()


# In[10]:


MS_columns=['funder','installer','subvillage','public_meeting','scheme_management','permit']


# In[11]:


dftrain=dftrain.drop('scheme_name', axis=1)
dftest=dftest.drop('scheme_name', axis=1)


# In[12]:


dftest[MS_columns].info()


# In[13]:


mode_train_public_meeting=dftrain['public_meeting'].mode()[0]
mode_train_scheme_management=dftrain['scheme_management'].mode()[0]
mode_train_permit=dftrain['permit'].mode()[0]


# In[14]:


dftrain['public_meeting']=dftrain['public_meeting'].fillna(mode_train_public_meeting)
dftrain['scheme_management']=dftrain['scheme_management'].fillna(mode_train_scheme_management)
dftrain['permit']=dftrain['permit'].fillna(mode_train_permit)


# In[15]:


dftest['public_meeting']=dftest['public_meeting'].fillna(mode_train_public_meeting)
dftest['scheme_management']=dftest['scheme_management'].fillna(mode_train_scheme_management)
dftest['permit']=dftest['permit'].fillna(mode_train_permit)


# In[16]:


Missingvalue='missing'
dftrain['funder']=dftrain['funder'].fillna(Missingvalue)
dftrain['installer']=dftrain['installer'].fillna(Missingvalue)
dftrain['subvillage']=dftrain['subvillage'].fillna(Missingvalue)


# In[17]:


dftest['funder']=dftest['funder'].fillna(Missingvalue)
dftest['installer']=dftest['installer'].fillna(Missingvalue)
dftest['subvillage']=dftest['subvillage'].fillna(Missingvalue)


# In[18]:


dftrain.info()


# In[ ]:





# In[19]:


column_object=[]
for col in dftrain.columns:
    if((dftrain[col].dtypes==object)or(dftrain[col].dtypes==bool)):
        column_object.append(col)
    else:
        print(col)        


# In[20]:


column_object=['funder','installer','wpt_name','basin','subvillage','region','lga','ward','public_meeting',
 'recorded_by','scheme_management','permit','extraction_type','extraction_type_group','extraction_type_class',
 'management','management_group','payment','payment_type','water_quality','quality_group',
 'quantity','quantity_group','source','source_type','source_class','waterpoint_type','waterpoint_type_group']


# In[21]:


#########################################


# In[22]:


from sklearn.preprocessing import OrdinalEncoder
enc=OrdinalEncoder()
X_train_feautres=enc.fit_transform(dftrain[column_object])


# In[23]:


df_train1=pd.DataFrame(X_train_feautres, columns=column_object)


# In[24]:


dftrain=dftrain.drop(labels=column_object, axis=1)
print(dftrain.shape)


# In[25]:


frames=[dftrain,df_train1]
df_train = pd.concat(frames,axis=1)
print(df_train.shape)


# In[26]:


X_test_feautres=enc.fit_transform(dftest[column_object])
df_test1=pd.DataFrame(X_test_feautres, columns=column_object)
print(df_test1.shape)


# In[27]:


dftest=dftest.drop(labels=column_object, axis=1)
print(dftest.shape)


# In[28]:


frames1=[dftest,df_test1]
df_test = pd.concat(frames1,axis=1)
print(df_test.shape)


# In[29]:


print(df_train.shape)
print(df_test.shape)


# In[33]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_train['status_group']=le.fit_transform(df_train['status_group'])


# In[31]:


df_test[df_test['construction_year']==0].head()


# In[32]:


df_test.head(2)


# In[ ]:





# In[33]:


import datetime
df_train['date_recorded'] = df_train['date_recorded'].astype('datetime64[ns]')
df_train['Month']=df_train.date_recorded.dt.month
df_train['DAY']=df_train.date_recorded.dt.day
df_train['YEAR']=df_train.date_recorded.dt.year


# In[34]:


df_train[['Month','DAY','YEAR']].info()


# In[35]:


df_test['date_recorded'] = df_test['date_recorded'].astype('datetime64[ns]')
df_test['Month']=df_test.date_recorded.dt.month
df_test['DAY']=df_test.date_recorded.dt.day
df_test['YEAR']=df_test.date_recorded.dt.year


# In[36]:


df_train=df_train.drop('date_recorded', axis=1)
df_test=df_test.drop('date_recorded', axis=1)


# In[37]:


df_train.shape


# In[38]:


counter=0
for i in range(0,len(df_train)):
    if(df_train['construction_year'][i]==0):
        inpdata=df_train['YEAR'][i]
        df_train['construction_year'][i]=inpdata
    else:
        counter=counter+1
        print(counter)


# In[39]:


counter1=0
for i in range(0,len(df_test)):
    if(df_test['construction_year'][i]==0):
        inpdata=df_test['YEAR'][i]
        df_test['construction_year'][i]=inpdata
    else:
        counter1=counter1+1
        print(counter1)


# In[40]:


df_train['GAP']=df_train['YEAR']-df_train['construction_year']
df_test['GAP']=df_test['YEAR']-df_test['construction_year']    


# In[41]:


df_train=df_train.drop('YEAR', axis=1)
df_test=df_test.drop('YEAR', axis=1)


# In[42]:


df_train.shape


# In[43]:


df_test.shape


# In[44]:


df_train.columns


# In[45]:


df_train.to_csv('df_train.csv',index=False)
df_test.to_csv('df_test.csv',index=False)


# In[4]:


df_train=pd.read_csv('df_train.csv')
df_test=pd.read_csv('df_test.csv')


# In[19]:


Predictors=['amount_tsh', 'gps_height', 'longitude', 'latitude','num_private', 'region_code', 'district_code', 'population',
'construction_year','funder', 'installer', 'wpt_name','basin', 'subvillage', 'region', 'lga', 'ward', 'public_meeting',
'recorded_by', 'scheme_management', 'permit', 'extraction_type','extraction_type_group', 'extraction_type_class', 'management',
'management_group', 'payment', 'payment_type', 'water_quality','quality_group', 'quantity', 'quantity_group', 'source', 'source_type',
'source_class', 'waterpoint_type', 'waterpoint_type_group', 'Month','DAY', 'GAP']

Target=['status_group']


X=df_train[Predictors].values
y=df_train[Target].values
X=np.abs(X)
y=np.array(y)


# In[110]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=7, random_state=56, shuffle=True)
for train_index, test_index in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf1=XGBClassifier(base_score=0.5, learning_rate=0.1, max_delta_step=0, max_depth=15,n_estimators=1500)
    #clf2=GradientBoostingClassifier(random_state=0,learning_rate=0.1,n_estimators=200,max_depth=15)
    #clf3 =RandomForestClassifier(n_estimators=500, random_state=1,max_depth=10)
    #Gv= VotingClassifier(estimators=[('XGB', clf1), ('LG', clf2), ('RF', clf3)], voting='hard')
    model=clf1.fit(X_train, y_train)
    predictions=model.predict(X_test)
    print('*'*50)
    print("The Classification report",metrics.classification_report(y_test,predictions))
    print("The Confusion report",metrics.confusion_matrix(y_test,predictions))
    print("The Accuracy",metrics.accuracy_score(y_test,predictions))


# In[9]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# In[168]:


params={
     'alpha':[0.45,0.5,0.6,1,1.5,2.5]
}


# In[ ]:





# In[ ]:





# In[ ]:





# In[141]:


metrics.SCORERS.keys()


# In[ ]:





# In[11]:


X_test=df_test[Predictors].values
X_test=np.abs(X_test)


# In[143]:


y_t=y[:14850]


# In[88]:



X_test = SkF.fit_transform(X_test, y_t)
print(X_test.shape)


# In[89]:


th_scale=RobustScaler()
X_test=th_scale.fit_transform(X_test)


# In[12]:


pred_test=model.predict(X_test)
sample_df=pd.DataFrame(dftest,columns=['id'])
sample_df['preds']=pred_test


# In[13]:


def target_mapping(inpdata):
    if(inpdata==0):
        return('functional')
    elif(inpdata==1):
        return('functional needs repair')
    else:
        return('non functional')
        


# In[14]:


sample_df['status_group']=sample_df['preds'].apply(target_mapping)


# In[15]:


sample_df=sample_df.drop('preds', axis=1)


# In[16]:


sample_df.to_csv('sample_df.csv', index=False)


# In[17]:


target.groupby('status_group').size().plot(kind='bar')


# In[18]:


sample_df.groupby('status_group').size().plot(kind='bar')


# In[ ]:




