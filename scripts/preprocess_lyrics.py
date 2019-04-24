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

def stem_and_count(song, stopwordsSet, stemmer):
    tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
    tokens = tokenizer.tokenize(song.lower())
    stems = []
    counter = Counter()
    for t in tokens:
        if (not t in stopwordsSet and len(t) > 2):
            stems.append(stemmer.stem(t))
    for stem in stems:
        counter[stem] += 1
    return counter

def init_gen(df, stop, stem):
    print("\tUsing all songs")
    for index, song in df[["INDEX","lyrics"]].iterrows():
        yield (index, stem_and_count(song=song["lyrics"], stopwordsSet=stop, stemmer=stem)) 

def main(vocab_size=2000):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', action='store', dest='outdir',
                        help='Output directory', required=True)
    parser.add_argument('-i', action='store', dest='infile',
                        help='Input CSV file with lyrics and labels', required=True)
    parser.add_argument('--songs-per-genre', action='store', default=10000,
                        dest='n_songs_per_genre', type=int,
                        help='Number of songs per genre')
    parsed_args = parser.parse_args()
    outdir = parsed_args.outdir
    infile = parsed_args.infile
    n_songs_per_genre = parsed_args.n_songs_per_genre
    print("Begin data cleaning with arguments: ", parsed_args)

    input_df = pd.read_csv(infile)
    input_df.rename(columns={'index':'INDEX'}, inplace=True)

    print("Step 0: Drop songs without lyrics. Subset by genre, and down/upsample")
    chosen_genres = ["Rock","R&B","Pop","Metal","Jazz","Indie","Hip-Hop","Folk","Electronic","Country"]
    input_df = input_df[(input_df.genre.isin(chosen_genres))].dropna(0, subset=["lyrics"])

    seed=0
    by_genre = []
    for g in chosen_genres:
        by_genre.append(input_df[(input_df.genre == g)].sample(n=n_songs_per_genre,replace=True,random_state=seed))
    input_df = pd.concat(by_genre)

    print("Step 1: Stopping and Stemming")
    stemmer = PorterStemmer()
    stopwordsSet = stopwords.words("english")

    print("Step 2: Add all the counters and identify the top K words")
    total_count = Counter()
    iteration = 0
    for idx, c in init_gen(input_df, stopwordsSet, stemmer):
        iteration += 1
        if (iteration % 1000 == 0):
            print("\titeration", iteration)
        total_count += c

    # Keep only the keys
    top_K = [k for k,v in total_count.most_common(vocab_size)]

    # NOTE - need to reset the generator because we already traversed it once
    print("Step 3: Convert each song to bag-of-words representation")
    top_K_counts = []
    iteration = 0
    for idx, c in init_gen(input_df, stopwordsSet, stemmer):
        iteration += 1
        if (iteration % 1000 == 0):
            print("\titeration", iteration)
        # lookup on an unused key just returns 0, like we want
        this_song = []
        this_song.append(idx)
        this_song.extend([c[item] for item in top_K])
        top_K_counts.append(this_song)

    print("Step 4: Convert to DataFrame")
    bag_of_words_df = pd.DataFrame(top_K_counts)
    column_headers = []
    column_headers.append("INDEX")
    column_headers.extend(top_K)
    bag_of_words_df.columns = column_headers # Fix column headers (TODO is this a safe approach?)

    print("Step 5: Save to pickle...")
    print("\tfull-bag-of-words pickle...")
    bag_of_words_df.to_pickle(os.path.join(outdir, "full-bag-of-words.pickle"))

    print("\tSave full labels for song_ids...")
    genre_to_number = {genre: i for i, genre in enumerate(chosen_genres)}

    print("\tgenre-number-mapping.pickle")
    with open(os.path.join(outdir, "genre-number-mapping.pickle"), "wb") as f:
        pickle.dump(genre_to_number, f)
    
    full_labels = input_df[["INDEX","genre"]].replace({"genre": genre_to_number})
    #full_labels = input_df.replace({"genre": genre_to_number})["genre"]

    print("\tfull-labels.pickle")
    full_labels.to_pickle(os.path.join(outdir,"full-labels.pickle"))

    print("Step 6: Test/Train Split and save files...")
    # TODO - use exactly 80%, not approximately 80%
    te_tr_split_ratio = 0.80
    msk = rand(len(full_labels)) < te_tr_split_ratio

    print("\ttrain-bag-of-words.pickle")
    bag_of_words_df[msk].to_pickle(os.path.join(outdir,"train-bag-of-words.pickle"))

    print("\ttrain-labels.pickle")
    full_labels[msk].to_pickle(os.path.join(outdir,"train-labels.pickle"))
    
    print("\ttest-bag-of-words.pickle")
    bag_of_words_df[~msk].to_pickle(os.path.join(outdir,"test-bag-of-words.pickle"))

    print("\ttest-labels.pickle")
    full_labels[~msk].to_pickle(os.path.join(outdir,"test-labels.pickle"))

if __name__ == "__main__":
    #main()
    main(vocab_size=5000)
