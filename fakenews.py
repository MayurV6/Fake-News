# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:59:36 2020

@author: Mayur Vishwasrao
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Embedding,LSTM,Dropout
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("D:/fakenews/fake-news/train.csv")
df.head()

# check null values
df.isnull().sum()

# drop null values from the dataset
df = df.dropna()

x = df.drop('label',axis = 1)
y = df['label']
 
# Input Data for Pre-processing
X_train = x.copy()

# reset index as we have removed null values from the dataset
X_train.reset_index(inplace=True)

# Download stopwords from nltk
nltk.download("stopwords")

# Pre-processing of text data using NLP techniqies like regex, stemming, etc.
ps = PorterStemmer()
x_train_corpus = []
for i in range(0,len(X_train)):
    cl_df = re.sub('[^a-zA-Z]',' ',X_train['title'][i])  
    cl_df = cl_df.lower()
    cl_df = cl_df.split()
    
    cl_df = [ps.stem(word) for word in cl_df if not word in stopwords.words("english")]
    cl_df = " ".join(cl_df)
    x_train_corpus.append(cl_df)
    
      
#Vocabulary Size
voc_size = 5000
onehot_train = [one_hot(words,voc_size) for words in x_train_corpus]
onehot_train
 
# Sentence length
sent_length = 20
embeded_x_train = pad_sequences(onehot_train,padding= 'pre',maxlen = sent_length)
embeded_x_train
 

embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length = sent_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu'))
model.compile(optimizer = 'adam',loss ='binary_crossentropy',metrics=['accuracy'])


x_final = np.array(embeded_x_train)
y_final = np.array(y)
x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size=0.2,random_state=42)

model.summary()

model_df = model.fit(x_train,y_train,validation_data =(x_test,y_test),batch_size=64,epochs=10)


y_pred = model.predict_classes(x_test)
accuracy_score(y_test,y_pred)

mat = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(mat,figsize=(6,6),show_normed=True)


# Testing on Test Data
Test = pd.read_csv("D:/fakenews/fake-news/test.csv") 
Test_id = Test["id"]

# Drop unnecessary clumns from the dataset
X_test = Test.drop(['text','id','author'],axis=1)

# check null values from the columns
X_test.isnull().sum()

# filling missing values with Fake text as submit file is of 5200 rows, so we can't remove the missing values
X_test.fillna('Fake',inplace=True)

X_test.shape

# pre-processing of text dataset
ps = PorterStemmer()
x_test_corpus = []
for i in range(0,len(X_test)):
    cl_df_test = re.sub('[^a-zA-Z]',' ',X_test['title'][i])  
    cl_df_test = cl_df_test.lower()
    cl_df_test = cl_df_test.split()
    
    cl_df_test = [ps.stem(word) for word in cl_df_test if not word in stopwords.words("english")]
    cl_df_test = " ".join(cl_df_test)
    x_test_corpus.append(cl_df_test)
    
    
#Vocabulary Size
voc_size = 5000
onehot_test = [one_hot(words,voc_size) for words in x_test_corpus]
onehot_test

sent_length = 20
embeded_x_test = pad_sequences(onehot_test,padding= 'pre',maxlen = sent_length)
embeded_x_test
  
# predict on test dataset 
x_test_final = np.array(embeded_x_test)
y_test_pred = model.predict_classes(x_test_final)

# apeend the prediction into new list and create new dataframe with columns "id", and "label"
val=[]
for i in y_test_pred:
    val.append(i[0])
    
    
submission = pd.DataFrame({'id':Test_id, 'label':val})
submission.shape

# write final dataframe to csv
submission.to_csv('D:/fakenews/fake-news/submit.csv',index=False)
