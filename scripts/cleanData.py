#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

import nltk
import string
from nltk.corpus import stopwords
import enchant

from nltk.tokenize import RegexpTokenizer

# Load data
dfC = pd.read_pickle("../data/lyrics_clean.pickle")
# Init assisting data
stemmer = PorterStemmer()
stopwordsSet = stopwords.words("english")
d = enchant.Dict("en_US")

def mapAndDict(song):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(song.lower())
    countDict = defaultdict(int)
    stems = map(lambda x: stemmer.stem(x),[t for t in tokens if (not t in stopwordsSet)]) 
    #unusual = list([t for t in tokens if (not d.check(t))])
    #print(unusual)
    for stem in stems:
        countDict[stem] += 1
    return countDict
    

countDicts = map(lambda s: mapAndDict(s), dfC["lyrics"])
fullFrame = pd.DataFrame(list(countDicts)).fillna(0)
fullFrame.to_pickle("../data/fullFrame.pickle")

sortedCounts = fullFrame.sum(axis=0).sort_values()




top2kWords = sortedCounts[-2000:].keys()




topBagofWords = fullFrame[top2kWords]




topBagofWords.to_pickle("../data/topBagofWords10k.pickle")






# def unusual_words(text):
#     text_vocab = set(w.lower() for w in text if w.isalpha())
#     english_vocab = set(w.lower() for w in nltk.corpus.words.words())
#     unusual = text_vocab - english_vocab
#     return sorted(unusual)

for k in sampleFrame.keys():
    print(k)




for k in sampleFrame.keys():
    print(k)





wordSet = set()
dfC['stemmed'] = pd.Series("", index=dfC.index)
for i in range(dfC.shape[0]):
    print("{}/{}".format(i+1, dfC.shape[0]))
    tokens = nltk.word_tokenize(dfC.iloc[i]["clean_lyrics"])
    stems = [stemmer.stem(w) for w in tokens]
    for s in stems:
        wordSet.add(s)
    dfC.at[i, "stemmed"] = " ".join(stems)
    




dfC




vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dfC["stemmed"]) 
print(X.toarray())
print(vectorizer.get_feature_names())




stemList = dfC["stemmed"].tolist()




len(stemList[0].split(" "))




vectorizer = CountVectorizer()
X = vectorizer.fit_transform([stemList[0]]) 




xArr = X.toarray()




xArr[0].sum()




len(np.unique(nltk.word_tokenize(stemList[0])))




len(vectorizer.get_feature_names())






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

vectors = map(vectorize, stemList)




for v in vectors:
    print(v)




len(vectorList)




vectorList




d1 = defaultdict(int)
d1["hello"] = 1
d2 = defaultdict(int)
d2["dan"] = 1
pd.DataFrame([d1,d2]).fillna(0)




help(enchant)






