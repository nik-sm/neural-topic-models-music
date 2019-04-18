#!/usr/bin/env python
# coding: utf-8

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from pipeline import newPipe
import numpy as np
from helpers import *
import os
from pipeline import newPipe

## Train on Song Corpus

# In[17]:

hParams = {"mode":"tf-idf", "max_features":100}
corpus = np.load("../data/corpus.npy")
leGenres = np.load("../data/leGenres.npy")


# In[18]:


songStringCorpus = [" ".join(song) for song in corpus]


# In[19]:


songVectorizer = TfidfVectorizer(stop_words="english", max_features=hParams["max_features"])
songTFIDF = songVectorizer.fit_transform(songStringCorpus)


# In[ ]:

expPath = makeExpDir()
expDict = newPipe(songTFIDF, leGenres)
np.save(os.path.join(expPath, "LogisticRegressionDict.npy"), expDict)
with open(os.path.join(expPath, "hParams.txt"), "w") as f:
    for k,v in hParams:
        f.write("{}:{}\n".format(k,v))
