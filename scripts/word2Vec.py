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
"mode":"word2Vec"
}

avgFeatures = np.load("../data/nik/word2VecAvgFeatures.npy")
sumFeatues = np.load("../data/nik/word2VecSumFeatures.npy")
allfeaturesLi = [avgFeatures, sumFeatues]
featuresDesc = ["average of word2Vecs", "sum of word2Vecs"]
allgenres = np.load("../data/nik/word2VecGenreLabels.npy")
if hParams["subsample"]=="all":
    featuresLi=allfeaturesLi
    genres = [allgenres, allgenres]
else:
    featuresLi=[]
    genres = []
    for f in allfeaturesLi:
        f,g = resample(f,
            allgenres,
            replace=False,
            n_samples=hParams["subsample"])
        featuresLi.append(f)
        genres.append(g)




# In[6]:
def runExp(features, genres, expNum):
    expPath = makeExpDir()

    experimentDict = newPipe(features, genres,
        iters=hParams["iters"],
        max_iter=hParams["max_iter"],
        regularization=hParams["regularization"],
        multi_class=hParams["multi_class"])

    np.save(os.path.join(expPath, "LogisticRegressionDict.npy"), experimentDict)
    hParams["featureDescription"] = featuresDesc[expNum]
    np.save(os.path.join(expPath, "hParams.npy"), hParams)
    with open(os.path.join(expPath, "hParams.txt"), "w") as f:
        for k,v in hParams.items():
            f.write("{}:{}\n".format(k,v))

Parallel(n_jobs=2)(delayed(runExp)(tup[0], tup[1], i) for i,tup in enumerate(zip(featuresLi, genres)))
