#!/usr/bin/env bash 
set -euxo pipefail 

# Pre-process songs 

## Input = lyrics.csv 
## Output = full-bag-of-words.pickle, train-bag-of-words.pickle test-bag-of-words.pickle, full-labels.pickle, test-labels.pickle, train-labels.pickle
## Maybe: stemmed_vocab.pickle (dictionary of stem:{set_of_input_words})
command -v python3 > /dev/null 2>&1 || { echo "Missing Python3 install" >&2 ; exit 1; }

INFILE="data/input/lyrics.csv"
OUTDIR_BASE="output"
PRODLDA_THETAS_FILE="theta_needsoftmax.pickle"
SCHOLAR_THETAS_FILE="theta.train.npz"
LABEL_FILE="output/bow/full-labels.pickle"
BOW_OUTDIR="output/bow"
#PLOTS_DIR="output/plots"
#ACCURACY_VS_DIM_RESULTS=${PLOTS_DIR}/"accuracy_vs_dimension.txt"
# CSV with:
# model,dimension,accuracy

# n=10000, k 

# Docker run with environment variables:
if [ -z $SONGS_PER_GENRE ]; then
  $SONGS_PER_GENRE=10
fi

if [ -z $N_TOPICS ]; then
  $N_TOPICS=20
fi

if [ 1 -eq 2 ] ; then
# Should be passed as environment variable
test -f ${INFILE} || { echo "Missing input file!" ; exit 1; }
test -d ${BOW_OUTDIR} || { echo "Making bow directory"; mkdir -p ${BOW_OUTDIR}; }
#test -d ${PLOTS_DIR} || { echo "Making plots directory"; mkdir -p ${PLOTS_DIR}; }
time python scripts/preprocess_lyrics.py -i data/input/lyrics.csv \
                                         -o ${BOW_OUTDIR} \
                                         --songs-per-genre ${SONGS_PER_GENRE}


# See "substring removal": https://www.tldp.org/LDP/abs/html/string-manipulation.html
# a="n10000k20"
# echo ${a##n*k}
#for PARAMS in n10000k10 n10000k20 n10000k50 n10000k100 n10000k300; do
# NOTE no comma

for PARAMS in n10000k10 n10000k20 ; do
	N_TOPICS=${PARAMS##n*k}

	OUT=${OUTDIR_BASE}/${PARAMS}/scholar
	test -d ${OUT} || { echo "making Scholar output dir"; mkdir -p ${OUT}; }
	# Run Scholar unsupervised (ScholarU)
	#time python scripts/scholar/run_scholar.py ${BOW_OUTDIR} \
	#																					 -o ${OUT} \
	#																					 --train-prefix full \
	#																					 -k ${N_TOPICS} \
	#																					 --epochs 40

	# Scholar features for classification
	#time python scripts/classify/logr_scholar.py --theta-file ${OUT}/${SCHOLAR_THETAS_FILE} \
	#																						 --label-file ${LABEL_FILE} \
	#																						 --output-dir ${OUT}


	# Run Scholar supervised
	OUT=${OUTDIR_BASE}/${PARAMS}/scholar_supervised
	time python scripts/scholar/run_scholar.py ${BOW_OUTDIR} \
																						 -o ${OUT} \
																						 --train-prefix train \
																						 --test-prefix "test" \
																						 --label genre \
																						 -k ${N_TOPICS} \
																						 --epochs 100


# Run Scholar supervised (ScholarS) (classification already performed)
# time python scripts/run_scholar.py data/bow/ --train-prefix train-bag-of-words --labels data/bow/labels.pickle

	# Run prodLDA
	#OUT=${OUTDIR_BASE}/${PARAMS}/prodlda
	#test -d ${OUT} || { echo "making prodLDA output dir"; mkdir -p ${OUT}; }
	#time python scripts/prodlda/tf_run.py -i ${BOW_OUTDIR}/full-bag-of-words.pickle \
	#																			-o ${OUT} \
	#																			-f 100 \
	#																			-s 100 \
	#																			-e 20 \
	#																			-r 0.002 \
	#																			-b 200 \
	#																			-k ${N_TOPICS}

# ProdLDA features for classification
	#time python scripts/classify/logr_prodlda.py --theta-file ${OUT}/${PRODLDA_THETAS_FILE} \
	#																						 --label-file ${LABEL_FILE} \
	#																						 --output-dir ${OUT}

done

# END OF NEURAL LDA MODELS


# BEGIN BASELINE MODELS


#time python scripts/tf_run.py -f 100 -s 100 -e 300 -r 0.002 -b 20 -t 50

exit 1
fi

# Generate TF/IDF Features
#time python scripts/tf_idf.py -i ${BOW_OUTDIR}/full-bag-of-words.pickle

# Word2vec + doc2vec preprocessing (lower, tokenize)

# Word2Vec

#test -d output/baseline || { echo "Making baseline directory"; mkdir -p output/baseline; }
#time python scripts/makeCorpusAndGenres.py --infile data/input/lyrics.csv \
#																					 --outdir output/baseline \
#																					 --songs-per-genre ${SONGS_PER_GENRE}


#test -d output/baseline/w2v_d2v || { echo "Making W2V and D2V directory"; mkdir -p output/baseline/w2v_d2v; }
#time python scripts/train_w2v_d2v.py --infile output/baseline/corpus.npy \
#                             --outdir output/baseline/w2v_d2v \
#														 --dimension 50


OUT=output/baseline/word2vec
test -d ${OUT} || { echo "Making W2V directory"; mkdir -p ${OUT}; }
time python scripts/classify/logr_word2vec.py --model output/baseline/w2v_d2v/word2vec.model \
                                              --corpus output/baseline/corpus.npy \
                                              --label-file output/baseline/genres.npy \
																							--output-dir ${OUT}

# Doc2Vec

OUT=output/baseline/doc2vec
test -d ${OUT} || { echo "Making W2V directory"; mkdir -p ${OUT}; }
time python scripts/classify/logr_doc2vec.py --model output/baseline/w2v_d2v/doc2vec.model \
                                             --label-file output/baseline/genres.npy \
                                  	--output-dir ${OUT}

# Run LogR on each of the following feature sets:

## ScholarU

## prodLDA

## TF/IDF

## "Raw" bag-of-words

## word2vec

## doc2vec
