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


# https://unix.stackexchange.com/questions/368246/cant-use-alias-in-script-even-if-i-define-it-just-above
if [ "$#" -eq 1 ]; then
	if [ "$1" == "dryrun" ]; then
		# Intercept all calls to `python` and echo instead
		set +x
		python () {
			/bin/echo python $@
		}
	fi
fi

# NOTE - docker pid always 1
#TIMESTAMP="$(date +"%Y_%m_%d")_$$"
#TIMESTAMP="$(date +"%Y_%m_%d_%s")"
TIMESTAMP="$(date +"%Y_%m_%d")"

echo "##################################"
echo "BEGIN PIPELINE"
echo "##################################"
# Pre-process songs 

# Display library versions
pip freeze

command -v python3 > /dev/null 2>&1 || { echo "Missing Python3 install" >&2 ; exit 1; }

INFILE="data/input/lyrics.csv"
PRODLDA_THETAS_FILE="theta_needsoftmax.pickle"
SCHOLAR_THETAS_FILE="theta.train.npz"
BOW_OUTDIR="output/bow"
FULL_LABEL_FILE="${BOW_OUTDIR}/full-labels.pickle"
FULL_BOW_FILE="${BOW_OUTDIR}/full-bag-of-words.pickle"

# Docker run with environment variables:
if [ -z $SONGS_PER_GENRE ]; then
  echo "Missing environment variable SONGS_PER_GENRE!"
	exit 1
fi

# See "substring removal": https://www.tldp.org/LDP/abs/html/string-manipulation.html
# a="n10000k20"
# echo ${a##n*k}
# NOTE no comma

for PARAMS in k300 k100 k50 k20 k10; do
	OUTDIR_BASE="output/${TIMESTAMP}"
	N_TOPICS=${PARAMS##k}
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
																							 --label-file ${FULL_LABEL_FILE} \
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
	time python scripts/prodlda/tf_run.py -i ${FULL_BOW_FILE} \
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
																							 --label-file ${FULL_LABEL_FILE} \
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
echo "BEGIN SEMI-SUPERVISED MODELS"
echo "##################################"

OUTDIR_BASE=output/${TIMESTAMP}/semi
ACCURACY_FILE=${OUTDIR_BASE}/accuracies.txt
test -d ${OUT} || { echo "making semi supervised output dir"; mkdir -p ${OUT}; }
for P in 0.2 0.5 0.8 1.0; do
	OUT=${OUTDIR_BASE}/${P}
	time python scripts/make-semi-labels.py --infile ${FULL_BOW_FILE} \
                                          --outdir ${OUT} \
																					--percent-supervise ${P}


	for reconstr_loss in 0.0 0.1 1.0 10.0; do 
		for kl_loss in     0.0 0.1 1.0 10.0; do  
			for classification_loss in 1.0; do 
				echo >> ${ACCURACY_FILE}
				echo >> ${ACCURACY_FILE}
				echo percent${P},recon${reconstr_loss},kl${kl_loss},cl${classification_loss} >> ${ACCURACY_FILE}

				OUT2=${OUTDIR_BASE}/${P}/recon${reconstr_loss}/kl${kl_loss}/cl${classification_loss}
				test -d ${OUT2} || { echo "making scholar semi supervised output dir"; mkdir -p ${OUT2}; }
				time python scripts/scholar/run_scholar.py ${BOW_OUTDIR} \
																									 -o ${OUT2} \
																									 --train-prefix semi-train \
																									 --test-prefix "test" \
																									 --label genre \
																									 -k ${N_TOPICS} \
																									 --epochs 90 \
																									 --reconstr-loss-coef ${reconstr_loss} \
																									 --kl_loss_coef ${kl_loss} \
																									 --classification_loss_coef ${classification_loss} \
			                                             --accuracy-file ${ACCURACY_FILE}

			done
		done
	done


done


echo "##################################"
echo "END PIPELINE"
echo "##################################"
