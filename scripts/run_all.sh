#!/usr/bin/env bash 
set -euxo pipefail 

# Pre-process songs 

## Input = lyrics.csv 
## Output = full-bag-of-words.pickle, train-bag-of-words.pickle test-bag-of-words.pickle, full-labels.pickle, test-labels.pickle, train-labels.pickle
## Maybe: stemmed_vocab.pickle (dictionary of stem:{set_of_input_words})
command -v python3 > /dev/null 2>&1 || { echo "Missing Python3 install" >&2 ; exit 1; }

INFILE="data/input/lyrics.csv"
PRODLDA_OUTDIR="output/prodlda"
PRODLDA_THETAS="output/prodlda/theta_needsoftmax.pickle"
SCHOLAR_OUTDIR="output/scholar"
SCHOLAR_THETAS="output/scholar/theta.train.npz"
LABEL_FILE="output/bow/full-labels.pickle"
BOW_OUTDIR="output/bow"

# Docker run with environment variables:
if [ -z $SONGS_PER_GENRE ]; then
  $SONGS_PER_GENRE=10
fi

if [ -z $N_TOPICS ]; then
  $N_TOPICS=20
fi

# Should be passed as environment variable
test -f ${INFILE} || { echo "Missing input file!" ; exit 1; }
test -d ${BOW_OUTDIR} || { echo "Making bow directory"; mkdir -p ${BOW_OUTDIR}; }
time python scripts/preprocess_lyrics.py -i data/input/lyrics.csv \
                                         -o ${BOW_OUTDIR} \
                                         --songs-per-genre ${SONGS_PER_GENRE}

# Run Scholar unsupervised (ScholarU)
time python scripts/scholar/run_scholar.py ${BOW_OUTDIR} \
                                           -o ${SCHOLAR_OUTDIR} \
                                           --train-prefix full-bag-of-words \
                                           -k ${N_TOPICS} \
                                           --epochs 20

# Scholar features for classification
time python scripts/classify/logr_scholar.py --theta-file ${SCHOLAR_THETAS} \
                                             --label-file ${LABEL_FILE}


# Run Scholar supervised (ScholarS) (classification already performed)
# time python scripts/run_scholar.py data/bow/ --train-prefix train-bag-of-words --labels data/bow/labels.pickle

# Run prodLDA

test -d ${PRODLDA_OUTDIR} || { echo "making prodLDA output dir"; mkdir -p ${PRODLDA_OUTDIR}; }
#time python scripts/prodlda/pytorch_run.py -i ${BOW_OUTDIR}/full-bag-of-words.pickle \
#                                           -f 100 \
#                                           -s 100 \
#                                           -e 20 \
#                                           -r 0.002 \
#                                           -b 200 \
#                                           -k ${N_TOPICS}


time python scripts/prodlda/tf_run.py -i ${BOW_OUTDIR}/full-bag-of-words.pickle \
                                      -o ${PRODLDA_OUTDIR} \
                                      -f 100 \
                                      -s 100 \
                                      -e 20 \
                                      -r 0.002 \
                                      -b 200 \
                                      -k ${N_TOPICS}

# ProdLDA features for classification
time python scripts/classify/logr_prodlda.py --theta-file ${PRODLDA_THETAS} \
                                             --label-file ${LABEL_FILE}


#time python scripts/tf_run.py -f 100 -s 100 -e 300 -r 0.002 -b 20 -t 50

# TODO - left off here 4/19 7pm
exit 1

# Generate TF/IDF Features
#time python scripts/tf_idf.py -i ${BOW_OUTDIR}/full-bag-of-words.pickle

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
