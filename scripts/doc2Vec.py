#!/usr/bin/env python
# coding: utf-8

# In[138]:


import numpy as np
import pandas as pd
import gensim.utils as gu
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# import nltk
# from nltk.stem.porter import *
# from sklearn.feature_extraction.text import CountVectorizer
# from collections import defaultdict
# import string
# from nltk.corpus import stopwords
# import enchant
# from nltk.tokenize import RegexpTokenizer
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
import os
import time
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

from pipeline import newPipe

# In[104]:

hParams = {"doc2VecSize": 300, "epochs":100, "min_count":2}


if not os.path.isfile("../data/docCorpus.npy"):
    print("creating doc corpus")
    df = pd.read_csv("../data/lyrics.csv").dropna(0, subset=["lyrics","genre"])

    # In[106]:


    df["tokens"] = "remove this row"
    df["tokens"] = df["tokens"].astype(str)


    # In[107]:


    badWords = ["verse", "chorus"]
    def doWork(song, songIDX):
        if not pd.isnull(song):
            return TaggedDocument([w for w in gu.simple_preprocess(song) if w not in badWords], [songIDX])

        


    # In[108]:


    corpus = Parallel(n_jobs=4)(delayed(doWork)(song, i) for i, song in enumerate(df["lyrics"].dropna()))
    np.save("../data/docCorpus.npy", corpus)
    stop
else:
    print("loading doc corpus")
    corpusLi = np.load("../data/docCorpus.npy")
    corpus = [TaggedDocument(c[0], c[1]) for c in corpusLi]
# In[139]:


leGenres = np.load("../data/leGenres.npy")


# In[111]:


model = Doc2Vec(vector_size=hParams["doc2VecSize"],
                min_count=hParams["min_count"],
                epochs=hParams["epochs"])


# In[112]:


model.build_vocab(corpus)


# In[113]:


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


# In[114]:


model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[EpochLogger()])


# In[119]:


d2VFeatures = np.zeros((len(corpus), hParams["doc2VecSize"]))
for i in range(len(corpus)):
    d2VFeatures[i] = model.infer_vector(corpus[i].words)


expPath = makeExpDir()

expDict = newPipe(d2VFeatures, leGenres)
np.save(os.path.join(expPath, "LogisticRegressionDict.npy"), expDict)
with open(os.path.join(expPath, "hParams.txt"), "w") as f:
    for k,v in hParams:
        f.write("{}:{}\n".format(k,v))


