from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-o', action='store', dest='outdir',
    #                    help='Output directory', required=True)
    parser.add_argument('--theta-file', action='store', help='Trained thetas for each song', required=True)
    parser.add_argument('--label-file', action='store', help='Genre label for each song', required=True)
    parser.add_argument('--output-dir', action='store', help='Output directory', required=True)
    args = parser.parse_args()
    print("Running logr_scholar with arguments: ", args)

    full_data=pd.read_pickle(args.theta_file).values
    full_labels=pd.read_pickle(args.label_file)['genre'].values
    data_train,data_test,labels_train,labels_test = train_test_split(full_data, full_labels, test_size=0.2)

    clf = LogisticRegression(random_state=0, solver='lbfgs',
                            multi_class='multinomial').fit(data_train, labels_train)
    with open(os.path.join(args.output_dir,"accuracy.txt"), 'w') as f:
        f.write(str(clf.score(data_test, labels_test)))
    weights = clf.coef_
    np.savetxt(os.path.join(args.output_dir,"weights.txt"), weights)


if __name__ == '__main__':
    main()
