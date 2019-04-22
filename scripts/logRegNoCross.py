import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
import os
import time
from collections import Counter
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
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


def noCrossPipe(features, labels, iters=10, max_iter=4000, regularization="l2", multi_class="multinomial"):
    experimentDict = {}
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
        if regularization == "l2":
            fullLogReg = LogisticRegression(max_iter=max_iter, penalty="l2", class_weight='balanced', solver='sag',multi_class='multinomial')
        elif regularization == "l1":
            fullLogReg = LogisticRegression(max_iter=max_iter, penalty="l1", class_weight='balanced', solver='sag',multi_class='multinomial')
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
        testPreds = np.argmax(testProbs, axis=1)
        testF1 = f1_score(yTest, testPreds, average="weighted", labels=np.unique(testPreds))
        dict_i["testF1"] = testF1
        experimentDict["iterDict"].append(dict_i)
    return experimentDict
