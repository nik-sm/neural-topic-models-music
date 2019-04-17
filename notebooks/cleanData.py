#!/usr/bin/env python

import pandas as pd
from collections import defaultdict, Counter
import string
import nltk
nltk_path = './nltk_packages'
# NLTK needs to download several packages
nltk.data.path.append(nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
nltk.download('punkt', download_dir=nltk_path)
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def stemAndCount(song, stopwordsSet, stemmer):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(song.lower())
    stems = []
    counter = Counter()
    for t in tokens:
        if (not t in stopwordsSet):
            stems.append(stemmer.stem(t))
    for stem in stems:
        counter[stem] += 1
    return counter
    
def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()
    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)

def vectorize(doc):
    features = defaultdict(int)
    for token in tokenize(doc):
        features[token] += 1
    return features

def init_gen(n_songs, df, stop, stem):
    if n_songs != None:
        print("\tOnly using first {} songs".format(n_songs))
        for index, song in df[["index","lyrics"]].head(n_songs).iterrows():
            yield (index, stemAndCount(song=song["lyrics"], stopwordsSet=stop, stemmer=stem)) 
    else:
        print("\tUsing all songs")
        for index, song in df[["index","lyrics"]].iterrows():
            yield (index, stemAndCount(song=song["lyrics"], stopwordsSet=stop, stemmer=stem)) 

def main(n_songs=None):
    print("Begin data cleaning")

    infile = "../data/lyrics.csv"
    print("Input data file: {}".format(infile))
    df = pd.read_csv(infile)

    print("Step 0: Drop songs without lyrics. Subset by genre, and down/upsample")
    chosen_genres = ['Rock','R&B','Pop','Metal','Jazz','Indie','Hip-Hop','Folk','Electronic','Country']
    df = df[(df.genre.isin(chosen_genres))].dropna(0, subset=['lyrics'])

    n_sample=10000
    seed=0
    by_genre = []
    for g in chosen_genres:
        by_genre.append(df[(df.genre == g)].sample(n=n_sample,replace=True,random_state=seed))
    df = pd.concat(by_genre)

    print("Step 1: Stopping and Stemming")
    stemmer = PorterStemmer()
    stopwordsSet = stopwords.words("english")

    print("Step 2: Add all the counters and identify the top K words")
    total_count = Counter()
    iteration = 0
    for idx, c in init_gen(n_songs, df, stopwordsSet, stemmer):
        iteration += 1
        if (iteration % 1000 == 0):
            print("\titeration", iteration)
        total_count += c

    # Keep only the keys
    top_K = [k for k,v in total_count.most_common(2000)]

    # NOTE - need to reset the generator because we already traversed it once
    print("Step 3: Convert each song to bag-of-words representation")
    top_K_counts = []
    iteration = 0
    for idx, c in init_gen(n_songs, df, stopwordsSet, stemmer):
        iteration += 1
        if (iteration % 1000 == 0):
            print("\titeration", iteration)
        # lookup on an unused key just returns 0, like we want
        this_song = []
        this_song.append(idx)
        this_song.extend([c[item] for item in top_K])
        top_K_counts.append(this_song)

    print("Step 4: Convert to DataFrame")
    sampleFrame = pd.DataFrame(top_K_counts)
    column_headers = []
    column_headers.append("index")
    column_headers.extend(top_K)
    sampleFrame.columns = column_headers # Fix column headers (TODO is this a safe approach?)

    print("\tSave DF to pickle...")
    sampleFrame.to_pickle("../data/bag_of_words.pickle")

    print("\tSave all labels for song_ids...")
    df[(df.index.isin(sampleFrame['index']))][['index','song','year','artist','genre']].to_pickle("../data/song_labels.pickle")

if __name__ == '__main__':
    main()
    # NOTE the operation of taking only 100 happens after upsampling, so then the songs can be duplicated and the final "labels" file can have more rows than the data file.
    # So treat this n_songs as for debugging purposes only
    #main(n_songs=100)
