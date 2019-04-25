#!/usr/bin/env python
import argparse
import pickle
import os
import pandas as pd
from numpy.random import rand
from collections import Counter
import string
import nltk
nltk_path = "./data/nltk_packages"
# NLTK needs to download several packages
nltk.data.path.append(nltk_path)
nltk.download("stopwords", download_dir=nltk_path)
#nltk.download("punkt", download_dir=nltk_path)
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def stem_and_count(song, stopwordsSet, stemmer, doStem):
    tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
    tokens = tokenizer.tokenize(song.lower())
    words = []
    counter = Counter()
    for t in tokens:
        if (not t in stopwordsSet and len(t) > 2):
            if doStem:
                words.append(stemmer.stem(t))
            else:
                words.append(t)
    for word in words:
        counter[word] += 1
    return counter

def init_gen(df, stop, stem):
    print("\tUsing all songs")
    #import pdb; pdb.set_trace()
    #print(dir(df[["index","genre","lyrics"]].iterrows()))

#    count=0
    for index, series in df[["index","genre","lyrics"]].iterrows():
#        count += 1
#        if count == 10000:
#            break
        genre = series['genre']
        lyrics = series['lyrics']
        #print("test: {}, {}".format(genre, lyrics))
        yield (index, genre, stem_and_count(lyrics, stopwordsSet=stop, stemmer=stem, doStem=False)) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', action='store', help='Output directory', required=True)
    parser.add_argument('-i', '--infile', action='store', help='Input CSV file with lyrics and labels', required=True)
    parser.add_argument('--songs-per-genre', action='store', type=int, help='Number of songs per genre', required=True)
    parser.add_argument('--vocab-size', action='store', type=int, help='Number of top words to use for BoW representation', required=True)
    args = parser.parse_args()
    
    print("Begin data cleaning with arguments: ", args)

    input_df = pd.read_csv(args.infile)
    #input_df.rename(columns={'index':'index'}, inplace=True)
    # NO!

    print("Step 0: Drop songs without lyrics. Subset by genre")
    chosen_genres = ["Rock","R&B","Pop","Metal","Jazz","Indie","Hip-Hop","Folk","Electronic","Country"]
    input_df = input_df[(input_df.genre.isin(chosen_genres))].dropna(0, subset=["lyrics"])

    print("Step 1: Stopping and Stemming")
    stemmer = PorterStemmer()
    stopwordsSet = stopwords.words("english")

    print("Step 2: Add all the counters and identify the top K words")
    total_count = Counter()
    iteration = 0

    for _, _, c in init_gen(input_df, stopwordsSet, stemmer):
        iteration += 1
        if (iteration % 1000 == 0):
            print("\titeration", iteration)
        total_count += c

    # Keep only the keys
    top_K = [k for k,v in total_count.most_common(args.vocab_size)]

    # NOTE - need to reset the generator because we already traversed it once
    print("Step 3: Convert each song to bag-of-words representation")
    top_K_counts = []
    iteration = 0
    for idx, genre, c in init_gen(input_df, stopwordsSet, stemmer):
        iteration += 1
        if (iteration % 1000 == 0):
            print("\titeration", iteration)
        # lookup on an unused key just returns 0, like we want
        this_song = []
        this_song.append(idx)
        this_song.append(genre)
        this_song.extend([c[item] for item in top_K])
        top_K_counts.append(this_song)

    del(input_df) # save memory

    print("Step 4: Convert to DataFrame")
    bag_of_words_df = pd.DataFrame(top_K_counts)
    column_headers = []
    column_headers.append("index")
    column_headers.append("genre")
    column_headers.extend(top_K)
    bag_of_words_df.columns = column_headers # Fix column headers (TODO is this a safe approach?)
    bag_of_words_df.index = bag_of_words_df['index']

    # At this point, bag_of_words_df contains a header like this:
    # index genre word1 word1 ...

    print("Step 5: Resample for full unsupervised data")
    all_columns = bag_of_words_df.columns
    counts_cols_only = ["index"]
    counts_cols_only.extend(all_columns[2:])

    seed=0
    by_genre = []
    for g in chosen_genres:
        by_genre.append(bag_of_words_df.loc[bag_of_words_df['genre'] == g].sample(n=args.songs_per_genre,replace=True,random_state=seed))
    upsampled = pd.concat(by_genre)
    del(by_genre)

    print("Step 6: Save full-size data, labels, and the genre-number mapping to pickle...")
    print("\tfull-bag-of-words pickle...")
    upsampled[counts_cols_only].to_pickle(os.path.join(args.outdir, "full-bag-of-words.pickle"))

    genre_to_number = {genre: i for i, genre in enumerate(chosen_genres)}
    print("\tfull-labels.pickle")
    upsampled[["index","genre"]].replace({"genre": genre_to_number}).to_pickle(os.path.join(args.outdir,"full-labels.pickle"))

    print("\tgenre-number-mapping.pickle")
    with open(os.path.join(args.outdir, "genre-number-mapping.pickle"), "wb") as f:
        pickle.dump(genre_to_number , f)

    del(upsampled)

    print("Step 7: Train/Test split, ...")

    print("\tsave test data first")
    # TODO - use exactly 80%, not approximately 80%
    te_tr_split_ratio = 0.80
    msk = rand(len(bag_of_words_df)) < te_tr_split_ratio

    print("\ttest-bag-of-words.pickle")
    bag_of_words_df[counts_cols_only][~msk].to_pickle(os.path.join(args.outdir,"test-bag-of-words.pickle"))

    print("\ttest-labels.pickle")
    bag_of_words_df[["index", "genre"]][~msk].replace({"genre": genre_to_number}).to_pickle(os.path.join(args.outdir,"test-labels.pickle"))

    print("\tresample train data")
    seed=1
    by_genre = []
    for g in chosen_genres:
        by_genre.append(bag_of_words_df.loc[msk][bag_of_words_df[msk]['genre'] == g].sample(n=args.songs_per_genre,replace=True,random_state=seed))
    upsampled = pd.concat(by_genre)
    del(by_genre)
    del(bag_of_words_df)

    print("\ttrain-bag-of-words.pickle")
    upsampled[counts_cols_only].to_pickle(os.path.join(args.outdir,"train-bag-of-words.pickle"))

    print("\ttrain-labels.pickle")
    upsampled[["index","genre"]].replace({"genre": genre_to_number}).to_pickle(os.path.join(args.outdir,"train-labels.pickle"))

if __name__ == "__main__":
    main()
