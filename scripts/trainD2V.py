from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from helpers import *
import numpy as np

hParams = {"doc2VecSize": 300,
"epochs":100,
"min_count":2}

corpus = np.load("../data/nik/docCorpus.npy")
corpus = [TaggedDocument(c[0], c[1]) for c in corpus]

model = Doc2Vec(vector_size=hParams["doc2VecSize"],
                min_count=hParams["min_count"],
                epochs=hParams["epochs"])
model.build_vocab(corpus)

model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[EpochLogger()])
model.save("../models/doc2Vec.model")

# d2VFeatures = np.zeros((len(corpus), hParams["doc2VecSize"]))
# for i in range(len(corpus)):
#     d2VFeatures[i] = model.infer_vector(corpus[i].words)

# np.save("../data/nik/doc2VecFeatures.npy")