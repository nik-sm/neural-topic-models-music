import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

d2vF = np.load("../data/nik/d2VFeatures.npy")

X_embedded = TSNE(n_components=2).fit_transform(d2vF)

np.save("X_embedded")