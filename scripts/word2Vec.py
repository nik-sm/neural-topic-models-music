#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import gensim.utils as gu
from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
import cProfile
import re
from io import  StringIO
import pstats
import argparse
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.utils import resample
from time import gmtime, strftime
import os
from pipeline import newPipe
from helpers import *



corpus = np.load("../data/corpus.npy")
genres = np.load("../data/leGenres.npy")

hParams = {"word2VecSize":300,
"window":5,
"min_count":2,
"workers":4,
"epochs":50
}
# In[6]:


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


# ## Train Word2Vec Model

# In[8]:


model = Word2Vec(corpus, size=hParams["word2VecSize"], window=hParams["window"], min_count=hParams["min_count"], workers=hParams["workers"] , callbacks=[EpochLogger()])


# In[9]:


model.save("../data/word2Vec.model")


# In[7]:


model = Word2Vec.load("../data/word2Vec.model")


# In[11]:


model.train(corpus, total_examples=model.corpus_count, epochs=hParams["epochs"], callbacks=[EpochLogger()])


vecSums = []
vecAvg = []
for song in corpus:
    assert len(song), song
    vec = model[song[0]]
    allVecs = [model[song[0]]]
    for wordIDX in range(1,len(song)):
        vec = vec + model[song[wordIDX]]
        allVecs.append(model[song[wordIDX]])
    vecSums.append(vec)
    arr= np.array(allVecs)
    vecAvg.append(np.mean(arr, axis=1))

corpusTFIDF = np.array(vecAvg)
assert corpusTFIDF.shape[0] == len(corpus)
assert corpusTFIDF.shape[1] == word2VecSize
assert leGenres.shape[0] == len(corpus)

expPath = makeExpDir()

experimentDict = newPipe(corpusTFIDF, genres)

np.save(os.path.join(expPath, "LogisticRegressionDict.npy"), experimentDict)
with open(os.path.join(expPath, "hParams.txt"), "w") as f:
    for k,v in hParams:
        f.write("{}:{}\n".format(k,v))




