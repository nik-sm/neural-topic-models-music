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
    parser.add_argument('--train-bow', action='store', help='train bag-of-words file', required=True)
    parser.add_argument('--train-labels', action='store', help='train labels file', required=True)
    parser.add_argument('--test-bow', action='store', help='test bag-of-words file', required=True)
    parser.add_argument('--test-labels', action='store', help='test labels file', required=True)
    #parser.add_argument('--rank', action='store', help='Reduce rank tfidf matrix to this value', default=0, type=int, required=False)
    #parser.add_argument('--output-dir', action='store', help='Output directory', required=True)
    args = parser.parse_args()
    print("Running logr_tfidf with arguments: ", args)

    train_bow=np.load(args.train_bow, allow_pickle=True).values[:, 1:] # drop index column
    test_bow=np.load(args.test_bow, allow_pickle=True).values[:, 1:] # drop index column

    N_train_documents = len(train_bow)
    train_document_lengths = train_bow.sum(axis=1)[:, None] # broadcast this column vector for each word
    scaled_train_tf = train_bow / (1 + train_document_lengths)

    test_document_lengths = test_bow.sum(axis=1)[:, None] # broadcast this column vector for each word
    scaled_test_tf = test_bow / (1 + test_document_lengths)


    # NOTE - This will get used to make IDF for BOTH train and test
    train_df = (train_bow > 0).sum(axis=0) 
    train_idf = np.log(N_train_documents / (1 + train_df))[ None, :] # broadcast this vector for each row

    train_tfidf = scaled_train_tf * train_idf
    print("train_tfidf.shape", train_tfidf.shape)

    test_tfidf = scaled_test_tf * train_idf # NOTE use of train_idf here
    print("test_tfidf.shape", test_tfidf.shape)

    train_labels=np.load(args.train_labels, allow_pickle=True).values[:,1] # keep only the label column
    print("train_labels.shape", train_labels.shape)
    test_labels=np.load(args.test_labels, allow_pickle=True).values[:,1] # keep only the label column
    print("test_labels.shape", test_labels.shape)

    

    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(train_tfidf, train_labels)

    accuracy=str(clf.score(test_tfidf, test_labels))
    print("TF-IDF-full ACCURACY: ", accuracy)
    print("tf-idf-full,{}".format(accuracy))


    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(train_bow, train_labels)

    accuracy=str(clf.score(test_bow, test_labels))
    print("BOW-full unscaled ACCURACY: ", accuracy)
    print("bow-full-unscaled,{}".format(accuracy))


    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(scaled_train_tf, train_labels)

    accuracy=str(clf.score(scaled_test_tf, test_labels))
    print("BOW scaled ACCURACY: ", accuracy)
    print("bow-full-scaled,{}".format(accuracy))



    for r in [10, 20, 50, 100, 300]:
        #if args.rank != 0:
        t0 = time()
        print("Begin svd")
        svd = TruncatedSVD(n_components=r, n_iter=7, random_state=42)
        svd.fit(train_tfidf)
        data_train_lo = svd.transform(train_tfidf)
        data_test_lo = svd.transform(test_tfidf)

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

        print("data_train_lo.shape", data_train_lo.shape)
        print("data_test_lo.shape", data_test_lo.shape)

        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial').fit(data_train_lo, labels_train)

        accuracy=str(clf.score(data_test_lo, labels_test))
        print("TF-IDF ACCURACY: ", accuracy)
        print("{},tf-idf-svd,{}".format(r,accuracy))


    #with open(os.path.join(args.output_dir,"accuracy.txt"), 'w') as f:
    #    f.write(accuracy)
    #weights = clf.coef_
    #np.savetxt(os.path.join(args.output_dir,"weights.txt"), weights)


if __name__ == '__main__':
    main()
