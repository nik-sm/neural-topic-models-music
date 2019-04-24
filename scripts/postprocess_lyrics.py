import numpy as np
import pandas as pd

semi = pd.read_pickle('output/bow/train-labels.pickle')
new_semi = semi.copy(deep=True)
new_semi['genre'] = -1


supervise_percent = 0.5
for g in range(10):
    g_subset_values = list(set(semi.loc[semi['genre'] == g]['INDEX'].values))
    l = len(g_subset_values)
    rand_ind = np.random.choice(g_subset_values, int((supervise_percent)*l), False)
    new_semi.loc[rand_ind, 'genre'] = g

new_semi.to_pickle('output/bow/semi-train-labels.pickle')
