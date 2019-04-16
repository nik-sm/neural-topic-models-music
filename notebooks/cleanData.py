#!/usr/bin/env python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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
        return (stemAndCount(song=song, stopwordsSet=stop, stemmer=stem) for song in df["lyrics"].head(n_songs))
    else:
        print("\tUsing all songs")
        return (stemAndCount(song=song, stopwordsSet=stop, stemmer=stem) for song in df["lyrics"])

def main(n_songs=None):
    print("Begin data cleaning")

    infile = "../data/lyrics_clean.pickle"
    print("Input data file: {}".format(infile))
    df = pd.read_pickle(infile)

    print("Step 1: Stopping and Stemming")
    stemmer = PorterStemmer()
    stopwordsSet = stopwords.words("english")

    print("Step 2: Add all the counters and identify the top K words")
    counters = init_gen(n_songs, df, stopwordsSet, stemmer)
    total_count = Counter()
    i = 0
    for c in counters:
        i += 1
        if (i % 1000 == 0):
            print("\titeration", i)
        total_count += c

    # Keep only the keys
    top_K = [k for k,v in total_count.most_common(2000)]

    # NOTE - need to reset the generator because we already traversed it once
    counters = init_gen(n_songs, df, stopwordsSet, stemmer)
    top_K_counts = []
    for c in counters:
        # lookup on an unused key just returns 0, like we want
        top_K_counts.append([c[item] for item in top_K])

    print("Step 3: Convert to DataFrame")
    sampleFrame = pd.DataFrame(top_K_counts)
    sampleFrame.columns = top_K # Fix column headers (TODO is this a safe approach?)

    print("\tSave DF to pickle...")
    sampleFrame.to_pickle("../data/bagOfWords.pickle")


    # TODO - below code needs more work
    print("Step 4: Word vectors (TODO)")
    wordSet = set()
    df['stemmed'] = pd.Series("", index=df.index)
    for i in range(df.shape[0]):
        if (i % 1000 == 0):
            print("{}/{}".format(i+1, df.shape[0]))
        tokens = nltk.word_tokenize(df.iloc[i]["clean_lyrics"])
        stems = [stemmer.stem(w) for w in tokens]
        for s in stems:
            wordSet.add(s)
        df.at[i, "stemmed"] = " ".join(stems)

    #vectorizer = CountVectorizer()
    #X = vectorizer.fit_transform(df["stemmed"]) 
    #print(X.toarray())
    #print(vectorizer.get_feature_names())

    stemList = df["stemmed"].tolist()


    #len(stemList[0].split(" "))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([stemList[0]]) 

    xArr = X.toarray()

    #xArr[0].sum()

    #len(np.unique(nltk.word_tokenize(stemList[0])))

    #len(vectorizer.get_feature_names())

    vectors = map(vectorize, stemList)

    #for v in vectors:
    #    print(v)


    #len(vectorList)

    #vectorList

if __name__ == '__main__':
    main()
    #main(n_songs=10000)
