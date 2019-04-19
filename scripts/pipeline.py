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


def newPipe(features, labels, iters=10, max_iter=4000, regularization="l2", multi_class="multinomial"):
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
                if multi_class=="multinomial":
                    if regularization == "l2":
                        logReg = LogisticRegression(max_iter=max_iter, penalty="l2", class_weight='balanced', C=c, solver='lbfgs',multi_class=multi_class)
                    elif regularization == "l1":
                        logReg = LogisticRegression(max_iter=max_iter, penalty="l1", class_weight='balanced', C=c, solver='lbfgs',multi_class=multi_class)
                    else:
                        assert False, "{} regularization is not supported".format(regularization)
                elif multi_class=="ovr":
                    if regularization == "l2":
                        logReg = LogisticRegression(max_iter=max_iter, solver="lbfgs", penalty="l2", class_weight='balanced', C=c,multi_class=multi_class)
                    elif regularization == "l1":
                        logReg = LogisticRegression(max_iter=max_iter, solver="lbfgs", penalty="l1", class_weight='balanced', C=c,multi_class=multi_class)
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
            fullLogReg = LogisticRegression(max_iter=max_iter, penalty="l2", class_weight='balanced', C=chosenC, solver='lbfgs',multi_class='multinomial')
        elif regularization == "l1":
            fullLogReg = LogisticRegression(max_iter=max_iter, penalty="l1", class_weight='balanced', C=chosenC, solver='lbfgs',multi_class='multinomial')
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
