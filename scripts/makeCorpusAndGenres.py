import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import gensim.utils as gu
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
import argparse


def doWork(i, lyrics, genre):
    if not i%1000:
        print("song number", i)
        #print("LYRICS: ", lyrics)
        #print("GENRE", genre)
    if not pd.isnull(lyrics):
        tokens = gu.simple_preprocess(lyrics)
        if tokens is not None and len(tokens) > 0:
            return tokens, genre

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', action='store', help='Output directory', required=True)
    parser.add_argument('-i', '--infile', action='store', help='Input CSV file with lyrics and labels', required=True)
    parser.add_argument('--songs-per-genre', action='store', type=int, help='Number of songs per genre', required=True)
    args = parser.parse_args()

    print("reading lyrics csv")
    chosen_genres = ["Rock","R&B","Pop","Metal","Jazz","Indie","Hip-Hop","Folk","Electronic","Country"]
    input_df = pd.read_csv(args.infile)
    input_df = input_df[(input_df.genre.isin(chosen_genres))].dropna(0, subset=["lyrics"])

    seed=0
    by_genre = []
    for g in chosen_genres:
        by_genre.append(input_df[(input_df.genre == g)].sample(n=args.songs_per_genre,replace=True,random_state=seed))
    input_df = pd.concat(by_genre)

# PREPARE Word list Corpus and Genres
    corpus_with_genres = Parallel(n_jobs=4)(delayed(doWork)(i, series['lyrics'], series['genre']) for (i, series) in input_df.iterrows())
    #print("separating corpus")
    #corpus = [t[0] for t in corpusAndGenres]
    #print("separating genres")
    #genres = df["genre"][[t[1] for t in corpusAndGenres]]
    #print("label encoding genres")
    #leGenres = LabelEncoder().fit_transform(genres)
    corpus = []
    genres = []
    for c in corpus_with_genres:
        if c is not None and len(c[0]) > 0:
            corpus.append(c[0])
            genres.append(c[1])
    print("saving files")
    np.save(os.path.join(args.outdir, "corpus.npy"), corpus)
    np.save(os.path.join(args.outdir, "genres.npy"), genres)
    #np.save(os.path.join(outdir,"../data/genres.npy"))
    #np.save(os.path.join(outdir,"../data/leGenres.npy"))

if __name__ == "__main__":
    main()
