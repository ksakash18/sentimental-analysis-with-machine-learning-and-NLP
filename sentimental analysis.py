#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk


# In[19]:


df=pd.read_csv("C:\\Users\\psykid\\Desktop\\sreedevi mam\\DATASETS\\IMDB Dataset.csv")
df


# In[27]:


vect=CountVectorizer()
docs=np.array(['hello akash here what parupadi','keep fighting','workout and focus parupadi'])
bag=vect.fit_transform(docs)


# In[28]:


print(vect.vocabulary_)


# In[29]:


print(bag.toarray())


# In[34]:


np.set_printoptions(precision=2)
tfidf=TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())


# In[36]:


nltk.download('stopwords')


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(use_idf=True,norm='l2',smooth_idf=True)
y=df.sentiment.values
x=tfidf.fit_transform(df['review'].values.astype('U'))


# In[38]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5)


# In[43]:


import pickle
from sklearn.linear_model import LogisticRegressionCV
clf=LogisticRegressionCV(cv=5,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=300).fit(x_train,y_train)



# In[47]:


saved_model=open('saved_model.sav','wb')
pickle.dump(clf,saved_model)
saved_model.close()


# In[48]:


filename='saved_model.sav'
saved_clf=pickle.load(open(filename,'rb'))
saved_clf.score(x_test,y_test)


# In[ ]:




