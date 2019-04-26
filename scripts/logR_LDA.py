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
"subsample":10000,
"mode":"tf-idf",
}


ldaF = np.load("../data/featuresForLDA/ldaFeatures.npy")
ldaL = np.load("../data/featuresForLDA/full-labels.pickle")["genre"]
if hParams["subsample"]=="all":
    features = ldaF
    genres = ldaL
else:
    features, genres = resample(ldaF,
        ldaL,
        replace=False,
        n_samples=hParams["subsample"])

# In[6]:
def runExp(features):
    expPath = makeExpDir()

    experimentDict = newPipe(features,
        genres,
        iters=hParams["iters"],
        max_iter=hParams["max_iter"],
        regularization=hParams["regularization"],
        multi_class=hParams["multi_class"])

    np.save(os.path.join(expPath, "LogisticRegressionDict.npy"), experimentDict)
    np.save(os.path.join(expPath, "hParams.npy"), hParams)
    with open(os.path.join(expPath, "hParams.txt"), "w") as f:
        for k,v in hParams.items():
            f.write("{}:{}\n".format(k,v))

runExp(features)
