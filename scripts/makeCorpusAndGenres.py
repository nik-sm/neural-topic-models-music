import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import gensim.utils as gu
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

print("reading lyrics csv")
df = pd.read_csv("../data/lyrics.csv").dropna(0, subset=["lyrics","genre"])


def doWork(song, songIDX):
    if not songIDX%100:
        print("song number", songIDX)
    if not pd.isnull(song):
        tokens = [w for w in gu.simple_preprocess(song) if w not in badWords]
        if len(tokens):
            return tokens, songIDX

# PREPARE Word list Corpus and Genres
print("doing work on songs")
corpusAndGenres = Parallel(n_jobs=4)(delayed(doWork)(song, i) for i, song in enumerate(df["lyrics"]))
print("separating corpus")
corpus = [t[0] for t in corpusAndGenres]
print("separating genres")
genres = df["genre"][[t[1] for t in corpusAndGenres]]
print("label encoding genres")
leGenres = LabelEncoder().fit_transform(genres)
print("saving files")
np.save("../data/corpus.npy")
np.save("../data/genres.npy")
np.save("../data/leGenres.npy")
<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
