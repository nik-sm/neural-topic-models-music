from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from pipeline import newPipe
import numpy as np
from helpers import *
import os
from pipeline import newPipe
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

hParams = {"mode":"tf-idf",
"max_features":2000,
"multi_class":"ovr",
"resample":"all",
"iters":10}

fullCorpus = np.load("../data/nik/listCorpus.npy")
fullGenres = np.load("../data/nik/genre.npy")

if hParams["resample"]=="all":
    corpus=fullCorpus
    leGenres = fullGenres
else:
    corpus, leGenres = resample(fullCorpus,
        fullGenres,
        replace=False,
        n_samples=hParams["resample"],
        )

songStringCorpus = [" ".join(song) for song in corpus]



songVectorizer = TfidfVectorizer(stop_words="english",
    max_features=hParams["max_features"])

songTFIDF = songVectorizer.fit_transform(songStringCorpus)
tfScaler = StandardScaler(with_mean=False)
songTFIDF = tfScaler.fit_transform(songTFIDF)

expPath = makeExpDir()
expDict = newPipe(songTFIDF, leGenres, iters=hParams["iters"], multi_class=hParams["multi_class"])
np.save(os.path.join(expPath, "LogisticRegressionDict.npy"), expDict)
with open(os.path.join(expPath, "hParams.txt"), "w") as f:
    for k,v in hParams.items():
        f.write("{}:{}\n".format(k,v))
