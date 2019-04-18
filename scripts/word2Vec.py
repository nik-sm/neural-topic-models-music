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


# In[2]:


df = pd.read_csv("../data/lyrics.csv").dropna(0, subset=["lyrics","genre"])


# In[44]:


badWords = ["verse", "chorus"]
def doWork(song, songIDX):
    if not pd.isnull(song):
        return [w for w in gu.simple_preprocess(song) if w not in badWords], songIDX


# In[45]:


corpusAndGenres = Parallel(n_jobs=4)(delayed(doWork)(song, i) for i, song in enumerate(df["lyrics"]))


# In[48]:


corpus = [t[0] for t in corpusAndGenres]


# In[50]:


genres = df["genre"][[t[1] for t in corpusAndGenres]]


# In[53]:


np.save("../data/corpus.npy", corpus)


# In[54]:


np.save("../data/genres.npy", genres)


# In[5]:


enc = OneHotEncoder()
oneHotGenres = enc.fit_transform(df["genre"].values.reshape((-1,1)))
oneHotGenres = oneHotGenres.toarray()
leGenres = LabelEncoder().fit_transform(df["genre"].values.reshape((-1,1)))


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


model = Word2Vec(corpus, size=300, window=5, min_count=1, workers=4, callbacks=[EpochLogger()])


# In[9]:


model.save("../data/word2Vec.model")


# In[23]:


model.doesnt_match("beyonce anye Jay-Z".split())


# In[7]:


model = Word2Vec.load("../data/word2Vec.model")


# In[11]:


model.train(corpus, total_examples=model.corpus_count, epochs=10, callbacks=[EpochLogger()])


# ## Evaluate

# In[32]:


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


def getMedianModel(li):
    med = np.median([l[1] for l in li])
    shifted = [np.abs(l[1] - med) for l in li]
    medianIDX = np.argmin(shifted)
    return li[medianIDX][0]


def newPipe(features, labels, iters=10, regularization="l2"):
    experimentDict = {}
    # Parameter Grid for hyper-parameter tuning
    paramGrid = {'C': np.logspace(-4, 4, num=10)}
    splits = 5 # Number of folds in Repeated Stratified K-Fold CV (RSKFCV)
    repeats = 5 # Number of repeats in Repeated Stratified K-Fold CV (RSKFCV)
    experimentDict["paramGrid"] = paramGrid
    experimentDict["RSKFCV splits"] = splits
    experimentDict["RSKFCV repeats"] = repeats
    experimentDict["regularization"] = regularization
    experimentDict["iterDict"] = []
    xTrainVal, xTestRaw, yTrainVal, yTestRaw = train_test_split(features, labels, test_size=0.2)
    for iteration in range(iters):
        print("iteration {} of {}".format(iteration, iters))
        dict_i = {}
        # store experiment information on first iteration
        if iteration == 0:
            experimentDict["xTrainVal"] = xTrainVal
            experimentDict["yTrainVal"] = yTrainVal
        rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats)
        # Store model perf on train and val data for model with each hyper-parameter assignment for all train/val splits
        trainRows = []
        valRows = []
        for train_index, validation_index in rskf.split(xTrainVal, yTrainVal):
            # Separate train and val data for a single run in Repeated Stratified K-Fold CV (RSKFCV)
            xTrain = xTrainVal[train_index]
            yTrain = yTrainVal[train_index]
            xVal = xTrainVal[validation_index]
            yVal = yTrainVal[validation_index]
            # Store performance in train and val data for each hyper-parameter assignment
            trainRow = []
            valRow = []
            for cNum, c in enumerate(paramGrid["C"]):
                if regularization == "l2":
                    logReg = LogisticRegression(penalty="l2", class_weight='balanced', C=c, solver='lbfgs',multi_class='multinomial')
                elif regularization == "l1":
                    logReg = LogisticRegression(penalty="l1", class_weight='balanced', C=c, solver='lbfgs',multi_class='multinomial')
                else:
                    assert False, "{} regularization is not supported".format(regularization)
                logReg.fit(xTrain, yTrain)
                trainProbs = logReg.predict_proba(xTrain)
                yPred = np.argmax(trainProbs, axis=1)
                trainF1 = f1_score(yTrain, yPred, average="weighted")
                valProbs = logReg.predict_proba(xVal)
                valPred = np.argmax(valProbs, axis=1)
                valF1 = f1_score(yVal, valPred, average="weighted")
                # store the performance for this c val on this train val split
                trainRow.append(trainF1)
                valRow.append(valF1)
            # store the performance for this train/val split
            valRows.append(valRow)
            trainRows.append(trainRow)
        # From results of RSKFCV figure out optimal c-value
        trainRows = np.array(trainRows)
        valRows = np.array(valRows)
        trainMean = np.mean(trainRows, axis=0)
        valMean = np.mean(valRows, axis=0)
        chosenCIDX = np.argmax(valMean)
        chosenC = paramGrid["C"][chosenCIDX]
        dict_i["chosen c value"] = chosenC
        dict_i["cv train f1"] = trainRows
        dict_i["cv val f1"] = valRows
        # Retrain model using all train and validation data using optimal C value
        if regularization == "l2":
            fullLogReg = LogisticRegression(penalty="l2", class_weight='balanced', C=chosenC, solver='lbfgs',multi_class='multinomial')
        elif regularization == "l1":
            fullLogReg = LogisticRegression(penalty="l1", class_weight='balanced', C=chosenC, solver='lbfgs',multi_class='multinomial')
        else:
            assert False, "{} regularization is not supported".format(regularization)
        fullLogReg.fit(xTrainVal, yTrainVal)
        dict_i["full model coefficients"] = fullLogReg.coef_
        dict_i["full model intercept"] = fullLogReg.intercept_
        dict_i["full model n_iter_"] = fullLogReg.n_iter_
        # get bootstrapped test set for this iteration
        xTest, yTest = resample(xTestRaw,yTestRaw, replace=True, random_state=iteration)
        dict_i["xTest"] = xTest
        dict_i["yTest"] = yTest
        # get predictions for test set
        testProbs = fullLogReg.predict_proba(xTest)
        dict_i["testProbs"] = testProbs
        # Calculate Test Performance
        testF1 = f1_score(yTest, np.argmax(testProbs, axis=1), average="weighted")
        dict_i["testF1"] = testF1
        experimentDict["iterDict"].append(dict_i)
    return experimentDict


# In[38]:


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


# In[40]:


for i, c in enumerate(corpus):
    assert len(c), df.iloc[i]


# In[ ]:




