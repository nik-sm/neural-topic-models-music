#!/usr/bin/env bash 
set -euxo pipefail 
# NOTE - (songs-per-genre) * (number of genres) must be >= batch_size. Otherwise script fails with:
	#Traceback (most recent call last):
	#  File "scripts/prodlda/tf_run.py", line 172, in <module>
	#    main()
	#  File "scripts/prodlda/tf_run.py", line 167, in main
	#    print_top_words(beta, vocab)
	#  File "scripts/prodlda/tf_run.py", line 83, in print_top_words
	#    for row in range(len(beta)):
	#TypeError: object of type 'int' has no len()




echo "##################################"
echo "BEGIN PIPELINE"
echo "##################################"
# Pre-process songs 

# Display library versions
pip freeze

## Input = lyrics.csv 
## Output = full-bag-of-words.pickle, train-bag-of-words.pickle test-bag-of-words.pickle, full-labels.pickle, test-labels.pickle, train-labels.pickle
## Maybe: stemmed_vocab.pickle (dictionary of stem:{set_of_input_words})
command -v python3 > /dev/null 2>&1 || { echo "Missing Python3 install" >&2 ; exit 1; }

INFILE="data/input/lyrics.csv"
PRODLDA_THETAS_FILE="theta_needsoftmax.pickle"
SCHOLAR_THETAS_FILE="theta.train.npz"
LABEL_FILE="output/bow/full-labels.pickle"
BOW_OUTDIR="output/bow"

# Docker run with environment variables:
if [ -z $SONGS_PER_GENRE ]; then
  $SONGS_PER_GENRE=10
fi
#if [ -z $N_TOPICS ]; then
#  $N_TOPICS=20
#fi

TIMESTAMP="$(date +"%Y_%m_%d")_$$"
echo ${TIMESTAMP}

echo "##################################"
echo "scripts/preprocess_lyrics.py"
echo "##################################"
test -f ${INFILE} || { echo "Missing input file!" ; exit 1; }
test -d ${BOW_OUTDIR} || { echo "Making bow directory"; mkdir -p ${BOW_OUTDIR}; }
#test -d ${PLOTS_DIR} || { echo "Making plots directory"; mkdir -p ${PLOTS_DIR}; }
time python scripts/preprocess_lyrics.py -i data/input/lyrics.csv \
                                         -o ${BOW_OUTDIR} \
                                         --songs-per-genre ${SONGS_PER_GENRE}

# See "substring removal": https://www.tldp.org/LDP/abs/html/string-manipulation.html
# a="n10000k20"
# echo ${a##n*k}
# NOTE no comma

for PARAMS in n10000k300 n10000k100 n10000k50 n10000k20 n10000k10; do
#for PARAMS in n10000k10 n10000k20; do
	OUTDIR_BASE="output/${TIMESTAMP}"
	N_TOPICS=${PARAMS##n*k}
	echo "##################################"
	echo "begin iteration. N_TOPICS=${N_TOPICS}"
	echo "##################################"


	echo "##################################"
	echo "scripts/scholar/run_scholar.py unsupervised"
	echo "##################################"
	OUT=${OUTDIR_BASE}/${PARAMS}/scholar
	test -d ${OUT} || { echo "making Scholar output dir"; mkdir -p ${OUT}; }
	# Run Scholar unsupervised (ScholarU)
	time python scripts/scholar/run_scholar.py ${BOW_OUTDIR} \
																						 -o ${OUT} \
																						 --train-prefix full \
																						 -k ${N_TOPICS} \
																						 --epochs 100


	echo "##################################"
	echo "scripts/classify/logr_scholar.py"
	echo "##################################"
	OUT=${OUTDIR_BASE}/${PARAMS}/logr_scholar
	test -d ${OUT} || { echo "making Scholar logr output dir"; mkdir -p ${OUT}; }
	time python scripts/classify/logr_scholar.py --theta-file ${OUTDIR_BASE}/${PARAMS}/scholar/${SCHOLAR_THETAS_FILE} \
																							 --label-file ${LABEL_FILE} \
																							 --output-dir ${OUT}


	echo "##################################"
	echo "scripts/scholar/run_scholar.py supervised"
	echo "##################################"
	OUT=${OUTDIR_BASE}/${PARAMS}/scholar_supervised
	test -d ${OUT} || { echo "making Scholar supervised output dir"; mkdir -p ${OUT}; }
	time python scripts/scholar/run_scholar.py ${BOW_OUTDIR} \
																						 -o ${OUT} \
																						 --train-prefix train \
																						 --test-prefix "test" \
																						 --label genre \
																						 -k ${N_TOPICS} \
																						 --epochs 100

	echo "##################################"
	echo "scripts/prodlda/tf_run.py"
	echo "##################################"
	OUT=${OUTDIR_BASE}/${PARAMS}/prodlda
	test -d ${OUT} || { echo "making prodLDA output dir"; mkdir -p ${OUT}; }
	time python scripts/prodlda/tf_run.py -i ${BOW_OUTDIR}/full-bag-of-words.pickle \
																				-o ${OUT} \
																				-f 100 \
																				-s 100 \
																				-e 100 \
																				-r 0.002 \
																				-b 200 \
																				-k ${N_TOPICS}

	echo "##################################"
	echo "scripts/classify/logr_prodlda.py"
	echo "##################################"
	OUT=${OUTDIR_BASE}/${PARAMS}/logr_prodlda
	test -d ${OUT} || { echo "making ProdLDA logr output dir"; mkdir -p ${OUT}; }
	time python scripts/classify/logr_prodlda.py --theta-file ${OUTDIR_BASE}/${PARAMS}/prodlda/${PRODLDA_THETAS_FILE} \
																							 --label-file ${LABEL_FILE} \
																							 --output-dir ${OUT}


# END OF NEURAL LDA MODELS


# BEGIN BASELINE MODELS
	echo "##################################"
	echo "scripts/makeCorpusAndGenres.py"
	echo "##################################"
	OUTDIR_BASE=output/${TIMESTAMP}/baseline
	OUT=${OUTDIR_BASE}/${PARAMS}

	test -d ${OUT} || { echo "Making baseline directory"; mkdir -p ${OUT}; }
	time python scripts/makeCorpusAndGenres.py --infile data/input/lyrics.csv \
																						 --outdir ${OUT} \
																						 --songs-per-genre ${SONGS_PER_GENRE}

	echo "##################################"
	echo "scripts/train_w2v_d2v.py"
	echo "##################################"
	OUT=${OUTDIR_BASE}/${PARAMS}/w2v_d2v
	test -d ${OUT} || { echo "Making W2V and D2V directory"; mkdir -p ${OUT}; }
	time python scripts/train_w2v_d2v.py --infile ${OUTDIR_BASE}/${PARAMS}/corpus.npy \
															 --outdir ${OUT} \
															 --dimension ${N_TOPICS}


	echo "##################################"
	echo "scripts/classify/logr_word2vec.py"
	echo "##################################"
	OUT=${OUTDIR_BASE}/${PARAMS}/logr_word2vec
	test -d ${OUT} || { echo "Making W2V directory"; mkdir -p ${OUT}; }
	time python scripts/classify/logr_word2vec.py --model ${OUTDIR_BASE}/${PARAMS}/w2v_d2v/word2vec.model \
																								--corpus ${OUTDIR_BASE}/${PARAMS}/corpus.npy \
																								--label-file ${OUTDIR_BASE}/${PARAMS}/genres.npy \
																								--output-dir ${OUT}

	echo "##################################"
	echo "scripts/classify/logr_doc2vec.py"
	echo "##################################"
	OUT=${OUTDIR_BASE}/${PARAMS}/logr_doc2vec
	test -d ${OUT} || { echo "Making W2V directory"; mkdir -p ${OUT}; }
	time python scripts/classify/logr_doc2vec.py --model ${OUTDIR_BASE}/${PARAMS}/w2v_d2v/doc2vec.model \
																							 --label-file ${OUTDIR_BASE}/${PARAMS}/genres.npy \
																							 --output-dir ${OUT}

done

echo "##################################"
echo "END PIPELINE"
echo "##################################"
# Generate TF/IDF Features
#time python scripts/tf_idf.py -i ${BOW_OUTDIR}/full-bag-of-words.pickle

# Run LogR on each of the following feature sets:

## ScholarU

## prodLDA

## TF/IDF

## "Raw" bag-of-words

## word2vec

## doc2vec
