#!/usr/bin/env bash 
set -euxo pipefail # Strict errors

# Pre-process songs 

## Input = lyrics.csv 
## Output = full-bag-of-words.pickle, train-bag-of-words.pickle test-bag-of-words.pickle, full-labels.pickle, test-labels.pickle, train-labels.pickle
## Maybe: stemmed_vocab.pickle (dictionary of stem:{set_of_input_words})
command -v python3 > /dev/null 2>&1 || { echo "Missing Python3 install" >&2 ; exit 1; }

# TODO python env. Docker?
# test -f py3/bin/activate || { echo "Making Python3 virtualenv"; python3 -m virtualenv py3; }
# source py3/bin/activate



test -d data/input || { echo "Missing input directory!" ; exit 1; }
test -d data/bow || { echo "Making bow directory"; mkdir -p data/bow; }
python scripts/preprocess_lyrics.py -i data/input/lyrics.csv -o data/bow/ --songs-per-genre 10

# Run Scholar unsupervised (ScholarU)
python scripts/scholar/run_scholar.py data/bow/ --train-prefix full-bag-of-words

# Run Scholar supervised (ScholarS) (classification already performed)
# python scripts/run_scholar.py data/bow/ --train-prefix train-bag-of-words --labels data/bow/labels.pickle

# Run prodLDA
test -d data/prodlda || { echo "making prodLDA output dir"; mkdir -p data/prodlda; }
python scripts/prodlda/tf_run.py -i data/bow/full-bag-of-words.pickle \
																 -o data/prodlda \
																 -f 100 \
																 -s 100 \
																 -e 100 \
																 -r 0.002 \
																 -b 50 \
																 -k 10
#python scripts/tf_run.py -f 100 -s 100 -e 300 -r 0.002 -b 20 -t 50


# TODO - left off here 4/19 7pm
exit 1

# Generate TF/IDF Features
python scripts/tf_idf.py -i /data/bow/full-bag-of-words.pickle


# Word2vec + doc2vec preprocessing (lower, tokenize)

# Word2Vec

# Doc2Vec


# Run LogR on each of the following feature sets:

## ScholarU

## prodLDA

## TF/IDF

## "Raw" bag-of-words

## word2vec

## doc2vec



## TODO
# ADAM instead of LBFGS
# https://github.com/xyzzzfred/scholar.git
