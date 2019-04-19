import numpy as np
import os
from pipeline import newPipe
from helpers import *
from joblib import Parallel, delayed


avgFeatures = np.load("../data/nik/word2VecAvgFeatures.npy")
sumFeatues = np.load("../data/nik/word2VecSumFeatures.npy")
featuresLi = [avgFeatures, sumFeatues]
genres = np.load("../data/nik/word2VecGenreLabels.npy")

hParams = {"word2VecSize":300,
"window":5,
"min_count":2,
"workers":4,
"epochs":50
}
# In[6]:
def runExp(features):
    expPath = makeExpDir()

    experimentDict = newPipe(features, genres)

    np.save(os.path.join(expPath, "LogisticRegressionDict.npy"), experimentDict)
    np.save(os.path.join(expPath, "hParams.npy"), hParams)
    with open(os.path.join(expPath, "hParams.txt"), "w") as f:
        for k,v in hParams:
            f.write("{}:{}\n".format(k,v))

Parallel(n_jobs=2)(delayed(runExp)(features) for features in featuresLi)
