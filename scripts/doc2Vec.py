import numpy as np
import os
from pipeline import newPipe
from helpers import *
from joblib import Parallel, delayed
from sklearn.utils import resample

hParams = {"iters":10,
"max_iter":4000,
"regularization":"l2",
"multi_class":"multinomial",
"subsample":1000
}

allfeatures = np.load("../data/nik/d2VFeatures.npy")
allgenres = np.load("../data/nik/genre.npy")
if hParams["subsample"]=="all":
    features = allfeatures
    genres = allgenres
else:
    features, genres = resample(allfeatures,
        allgenres,
        replace=False,
        n_samples=hParams["subsample"])

# In[6]:
def runExp(features):
    expPath = makeExpDir()

    experimentDict = newPipe(features,
        genres,
        iters=hParams[iters],
        max_iter=hParams["max_iter"],
        regularization=hParams["regularization"],
        multi_class=hParams["multi_class"])

    np.save(os.path.join(expPath, "LogisticRegressionDict.npy"), experimentDict)
    np.save(os.path.join(expPath, "hParams.npy"), hParams)
    with open(os.path.join(expPath, "hParams.txt"), "w") as f:
        for k,v in hParams:
            f.write("{}:{}\n".format(k,v))

runExp(features)

