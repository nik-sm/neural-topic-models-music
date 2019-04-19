import numpy as np
from helpers import *
from gensim.models.word2vec import Word2Vec

corpus = np.load("../data/nik/listCorpus.npy")
hParams = {"word2VecSize":300,
"window":5,
"min_count":2,
"workers":4,
"epochs":50
}

model = Word2Vec(corpus,
    size=hParams["word2VecSize"],
    window=hParams["window"],
    min_count=hParams["min_count"],
    workers=hParams["workers"] ,
    callbacks=[EpochLogger()])

model.save("../model/word2Vec.model")
model = Word2Vec.load("../model/word2Vec.model")
model.train(corpus, total_examples=model.corpus_count, epochs=hParams["epochs"], callbacks=[EpochLogger()])
model.save("../data/word2Vec.model")