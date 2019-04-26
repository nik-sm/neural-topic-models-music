from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os
from helpers import *
from gensim.models.word2vec import Word2Vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', help='word2vec model file', required=True)
    parser.add_argument('--corpus', action='store', help='list of list of words', required=True)
    parser.add_argument('--label-file', action='store', help='Genre label for each song', required=True)
    parser.add_argument('--output-dir', action='store', help='Output directory', required=True)
    args = parser.parse_args()
    print("Running logr_word2vec with arguments: ", args)

    model=Word2Vec.load(args.model)
    full_labels=np.load(args.label_file, allow_pickle=True)

    # get average word vector for each song

    n_total_words = 0
    n_dropped_words = 0
    corpus = np.load(args.corpus, allow_pickle=True)
    avg_vectors = []
    for i, song in enumerate(corpus):
        n_total_words += len(song)
        tmp = []
        song_length = len(song)
        n_song_dropped = 0
        for word in song:
            if word in model.wv.vocab:
                tmp.append(model.wv[word])
            else:
                n_song_dropped += 1
                n_dropped_words += 1
        if n_song_dropped == song_length:
            avg_vectors.append(np.zeros(model.vector_size))
        else:
            avg_vectors.append(np.mean(np.asanyarray(tmp), axis=0))

    print("total words: ", n_total_words)
    print("dropped words: ", n_dropped_words)

    data_train,data_test,labels_train,labels_test = train_test_split(avg_vectors, full_labels, test_size=0.2)

    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(data_train, labels_train)

    accuracy=str(clf.score(data_test, labels_test))
    print("WORD2VEC ACCURACY: ", accuracy)
    with open(os.path.join(args.output_dir,"accuracy.txt"), 'w') as f:
        f.write(accuracy)
    weights = clf.coef_
    np.savetxt(os.path.join(args.output_dir,"weights.txt"), weights)


if __name__ == '__main__':
    main()
