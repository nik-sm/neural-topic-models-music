import numpy as np
import pandas as pd
import argparse

def main():
    semi = pd.read_pickle('output/bow/train-labels.pickle')
    new_semi = semi.copy(deep=True)
    new_semi['genre'] = -1

    for g in range(10):
        g_subset_values = list(set(semi.loc[semi['genre'] == g]['index'].values))
        l = len(g_subset_values)
        rand_ind = np.random.choice(g_subset_values, int((args.percent_supervise)*l), False)
        new_semi.loc[rand_ind, 'genre'] = g

    new_semi.to_pickle('output/bow/semi-train-labels.pickle')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True, help='Input bag-of-words file')
    parser.add_argument('-o', '--outdir', required=True, help='Output directory')
    parser.add_argument('-p', '--percent-supervise' required=True, help='Percent supervision', type=float)
    parsed_args = parser.parse_args()

if __name__ == "__main__":
    main()
