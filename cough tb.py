#!/usr/bin/env python
# coding: utf-8

# In[145]:


import librosa
import random
import numpy as np


# In[146]:


# import scipy.io.wavfile
# sr, y = wavfile.read("train/audio/bed/00176480_nohash_0.wav")


# In[147]:


data_list = []
for i in range(1, 7):
    try:
        data, sampling_rate = librosa.load(f'music/{i}.wav')
#         data= data * 32767
    except:
        continue
#     print(data.shape)
#     print(len(data))
    data_list.append(data.tolist())
    
# data_list


# In[148]:


len_list = [len(e) for e in data_list]
# len_list
#max len to make equal size array
max_len = max(len_list)
# max_len

#equating len of lists
equal_len_data_list = [l + ([0] * (max_len - len(l))) if len(l) < max_len else l[: max_len] for l in data_list]
new_len = [len(e) for e in equal_len_data_list]
# new_len


# In[149]:


#list prop
# l1 = [2, 3]
# l2 = l1 + ([1] * 10)
# l2


# In[151]:


data_array = np.array(equal_len_data_list)
# data_array.shape
label_array = np.array([[1], [1], [0], [0], [0]])
label_array.shape


# In[152]:


#**********************************************Using SVM*********************************************************************# 
#svm is different from GNB in the sence that it looks at the degree relation between featers. 


# In[157]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix


# train test split 
X_train, X_test, y_train, y_test = train_test_split(features, labels,test_size = 0.1, random_state = 1)

# training vectorizer
# vectorizer = TfidfVectorizer()
# X_train = vectorizer.fit_transform(X_train)

# training the classifier 
svm = svm.SVC(C=1000)
svm.fit(X_train, y_train)

# testing against testing set 
# X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test) 
print('Confusion Matrix: ',confusion_matrix(y_test, y_pred))
print('Accuracy score: ',accuracy_score(y_test, y_pred))


# In[158]:


# test against new audio
def pred(audio):
    data, sampling_rate = librosa.load(f'music/{audio}')
    list_data = data.tolist()
#     print(len(list_data))
    audio = [ list_data + ([0] * (max_len - len(list_data))) if len(list_data) < max_len else list_data[:max_len]]
    prediction = svm.predict(audio)
    return prediction[0]


# In[159]:


pred('4.wav')


# In[160]:


for i in range(1, 7):
    try:
        print(pred(f'{i}.wav'))
        
    except:
        continue
        
    


# In[ ]:





# In[139]:


from sklearn.model_selection import train_test_split

features = data_array
labels = label_array
# Split our data
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)


# In[140]:


from sklearn.naive_bayes import GaussianNB


# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)


# In[141]:


# Make predictions
preds = gnb.predict(test)
print(preds)


# In[142]:


from sklearn.metrics import accuracy_score


# Evaluate accuracy
print(accuracy_score(test_labels, preds))


# In[143]:


# test against new audio
def pred(audio):
    data, sampling_rate = librosa.load(f'music/{audio}')
    list_data = data.tolist()
#     print(len(list_data))
    audio = [ list_data + ([0] * (max_len - len(list_data))) if len(list_data) < max_len else list_data[:max_len]]
    prediction = gnb.predict(audio)
    return prediction[0]


# In[144]:


for i in range(1, 7):
    try:
        print(pred(f'{i}.wav'))
        
    except:
        continue
        


# In[ ]:




