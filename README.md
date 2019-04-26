# Credits

Code adapted from [SCHOLAR](https://github.com/dallascard/scholar) and [ProdLDA](https://github.com/akashgit/autoencoding_vi_for_topic_models).

# Input 

The script expects a CSV input file of song data:

```bash
head -n1 data/lyrics.csv 
index,song,year,artist,genre,lyrics
```

# Pre-processing

To prepare the song lyrics for use in modeling, we:
- remove empty songs and unused genres
- lowercase
- stop
- stem
- up- and down-sample songs (with replacement) to obtain the same number of songs per genre

Preprocessing produces a number of files.

From the entire corpus:
-	`full-bag-of-words.pickle`
	- This contains an "index" column to map the song back to its labels
	- Each additional column is the number of occurrences of the word shown in the header
	- Each row represents one song, showing the count of each word.
-	`full-labels.pickle`
	- This associates the unique song numbers with other song labels
	- The header is: ["index", "genre"]
-	`genre-number-mapping.pickle`
	- This associates the numeric genre to the original genre string

From a train/test split of the data:
-	train-bag-of-words.pickle
-	train-labels.pickle
-	test-bag-of-words.pickle
-	test-labels.pickle

# How to run

## Build Docker Image
```bash
cd musicProject/
docker build -t music-pipeline .
```

## Run Docker Container
```bash
time docker run -e SONGS_PER_GENRE=30 -v $(pwd)/output/:/music/output/ music-pipeline
```

# TODO
- Update Docker pipeline
- Deduplicate files
- Incorporate final baselines and visualizations into pipeline
