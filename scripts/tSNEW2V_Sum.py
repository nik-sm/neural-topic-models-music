import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

sumFeatures = np.load("../data/nik/word2VecSumFeatures.npy")

X_embedded = TSNE(n_components=2).fit_transform(sumFeatures)

np.save("X_embedded")