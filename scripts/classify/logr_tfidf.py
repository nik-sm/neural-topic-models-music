from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import argparse
import os
from helpers import *
from gensim.models.word2vec import Word2Vec
from time import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', action='store', help='input bag-of-words file', required=True)
    parser.add_argument('--label-file', action='store', help='Genre label for each song', required=True)
    #parser.add_argument('--rank', action='store', help='Reduce rank tfidf matrix to this value', default=0, type=int, required=False)
    #parser.add_argument('--output-dir', action='store', help='Output directory', required=True)
    args = parser.parse_args()
    print("Running logr_word2vec with arguments: ", args)

    tf=np.load(args.infile, allow_pickle=True).values[:, 1:] # drop index column
    N_documents = len(tf)
    document_lengths = tf.sum(axis=1)[:, None] # broadcast this column vector for each word
    scaled_tf = tf / (1 + document_lengths)
    df = (tf > 0).sum(axis=0)
    idf = np.log(N_documents / (1 + df))[ None, :] # broadcast this vector for each row
    tfidf = scaled_tf * idf
    print("tfidf.shape", tfidf.shape)

    full_labels=np.load(args.label_file, allow_pickle=True).values[:,1] # keep only the label column
    print("full_labels.shape", full_labels.shape)

    data_train,data_test,labels_train,labels_test = train_test_split(tfidf, full_labels, test_size=0.2)

    
    #for r in [10, 20, 50, 100, 300]:
    #    #if args.rank != 0:
    #    t0 = time()
    #    print("Begin svd")
    #    svd = TruncatedSVD(n_components=r, n_iter=7, random_state=42)
    #    svd.fit(data_train)
    #    data_train_lo = svd.transform(data_train)
    #    data_test_lo = svd.transform(data_test)

    #    print("Rank {} SVD explained {} percent of variance in train data".format(r, svd.explained_variance_ratio_.sum()))
    #    #U, s, Vh = np.linalg.svd(data_train, full_matrices=False)
    #    #U_lo = U[:, 0:rank]
    #    #s_lo = s[0:rank, 0:rank]
    #    #data_train = np.matmul(U_lo, s_lo)
    #    #data_test = np.matmul(data_test, Vh.T[:,0:rank])
    #    #print("U.shape: ", U.shape)
    #    #print("s.shape: ", s.shape)
    #    #print("Vh.shape: ", Vh.shape)
    #    print("svd done in %0.3fs" % (time() - t0))

    #    print("data_train_lo.shape", data_train_lo.shape)
    #    print("data_test_lo.shape", data_test_lo.shape)

    #    clf = LogisticRegression(random_state=0, solver='lbfgs',
    #                             multi_class='multinomial').fit(data_train_lo, labels_train)

    #    accuracy=str(clf.score(data_test_lo, labels_test))
    #    print("TF-IDF ACCURACY: ", accuracy)
    #    print("{},tf-idf-svd,{}".format(r,accuracy))

    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(data_train, labels_train)

    accuracy=str(clf.score(data_test, labels_test))
    print("TF-IDF-full ACCURACY: ", accuracy)
    print("tf-idf-full,{}".format(accuracy))


    data_train,data_test,labels_train,labels_test = train_test_split(tf, full_labels, test_size=0.2)

    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(data_train, labels_train)

    accuracy=str(clf.score(data_test, labels_test))
    print("BOW-full ACCURACY: ", accuracy)
    print("bow-full,{}".format(accuracy))




    #with open(os.path.join(args.output_dir,"accuracy.txt"), 'w') as f:
    #    f.write(accuracy)
    #weights = clf.coef_
    #np.savetxt(os.path.join(args.output_dir,"weights.txt"), weights)


if __name__ == '__main__':
    main()
