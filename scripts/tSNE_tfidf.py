import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

allfeatures = np.load("../data/nik/tfidf.npy").item()

X_embedded = TSNE(n_components=2).fit_transform(allfeatures.toarray())

np.save("../results/tSNE_tfidf.npy")