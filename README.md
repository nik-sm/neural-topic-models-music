# Input 

The script expects a CSV input file of song data:

```bash
head -n1 data/lyrics.csv 
index,song,year,artist,genre,lyrics
```

# Output

TODO - this needs updating

Preprocessing produces 2 files:

- "bag_of_words.pickle" 
	- This contains a bag-of-words representation of song lyrics. 
	- 10,000 songs from the chosen set of genres are used (via up-sampling for small genres).
	- The columns are is "index" (unique song number), followed by one column per word in the top 2K words.
	- Each row represents one song, showing the count of each word.
- "song_labels.pickle"
	- This associates the unique song numbers with other song labels
	- The header is: ["index", "song", "year", "artist", "genre"]

# How to run

```bash
python3 -m virtualenv py3
source py3/bin/activate
pip install -r requirements
bash run_all.sh
```

# TODO

- cite sources for code!
- Confirm how thetas are extracted for prodLDA in tf_run.py and tf_model.py
- confirm that word counts and song labels are correct by taking a single song and tracking it from the end all the way back to the beginning
