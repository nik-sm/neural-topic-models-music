import numpy as np
import pandas as pd

df = pd.read_csv("../data/lyrics.csv").dropna(0, subset=["lyrics","genre"])

badWords = ["verse", "chorus"]
def doWork(song, songIDX):
    if not pd.isnull(song):
        tokens = [w for w in gu.simple_preprocess(song) if w not in badWords]
        if len(tokens):
            return [w for w in gu.simple_preprocess(song) if w not in badWords], songIDX

corpusAndGenres = Parallel(n_jobs=4)(delayed(doWork)(song, i) for i, song in enumerate(df["lyrics"]))

corpus = [t[0] for t in corpusAndGenres]

genres = df["genre"][[t[1] for t in corpusAndGenres]]

leGenres = LabelEncoder().fit_transform(genres)

np.save("../data/corpus.npy")
np.save("../data/genres.npy")
np.save("../data/leGenres.npy")