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

# Display library versions
pip freeze

command -v python3 > /dev/null 2>&1 || { echo "Missing Python3 install" >&2 ; exit 1; }

INFILE="data/input/lyrics.csv"
BOW_OUTDIR="output/bow"

# Docker run with environment variables:
if [ -z $SONGS_PER_GENRE ]; then
  echo "missing environment variable SONGS_PER_GENRE"
	exit 1
fi
if [ -z $VOCAB_SIZE ]; then
  echo "missing environment variable VOCAB_SIZE"
	exit 1
fi


echo "##################################"
echo "scripts/preprocess_lyrics.py"
echo "##################################"
test -f ${INFILE} || { echo "Missing input file!" ; exit 1; }
test -d ${BOW_OUTDIR} || { echo "Making bow directory"; mkdir -p ${BOW_OUTDIR}; }
#test -d ${PLOTS_DIR} || { echo "Making plots directory"; mkdir -p ${PLOTS_DIR}; }
time python scripts/preprocess_lyrics.py --infile data/input/lyrics.csv \
                                         --outdir ${BOW_OUTDIR} \
                                         --songs-per-genre ${SONGS_PER_GENRE} \
																				 --vocab-size ${VOCAB_SIZE}
