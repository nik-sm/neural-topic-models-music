import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

avgFeatures = np.load("../data/nik/word2VecAvgFeatures.npy")

X_embedded = TSNE(n_components=2).fit_transform(avgFeatures)

np.save("../results/tSNE_w2v_avg.npy")