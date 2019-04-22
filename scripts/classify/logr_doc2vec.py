from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os
from helpers import *
from gensim.models.doc2vec import Doc2Vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', help='doc2vec model file', required=True)
    parser.add_argument('--label-file', action='store', help='Genre label for each song', required=True)
    parser.add_argument('--output-dir', action='store', help='Output directory', required=True)
    args = parser.parse_args()
    print("Running logr_doc2vec with arguments: ", args)

    model=Doc2Vec.load(args.model)
    full_labels=np.load(args.label_file, allow_pickle=True)

    doc_vectors = []
    for i in range(len(full_labels)):
        doc_vectors.append(model.docvecs[i])

    data_train,data_test,labels_train,labels_test = train_test_split(doc_vectors, full_labels, test_size=0.2)

    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(data_train, labels_train)

    accuracy=str(clf.score(data_test, labels_test))
    print("DOC2VEC ACCURACY: ", accuracy)
    with open(os.path.join(args.output_dir,"accuracy.txt"), 'w') as f:
        f.write(accuracy)
    weights = clf.coef_
    np.savetxt(os.path.join(args.output_dir,"weights.txt"), weights)


if __name__ == '__main__':
    main()
