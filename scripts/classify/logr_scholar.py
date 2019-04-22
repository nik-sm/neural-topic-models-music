import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theta-file', action='store', help='Trained thetas for each song', required=True)
    parser.add_argument('--label-file', action='store', help='Genre label for each song', required=True)
    parser.add_argument('--output-dir', action='store', help='Output directory', required=True)
    args = parser.parse_args()
    print("Running logr_scholar with arguments: ", args)

    thetas = np.load(args.theta_file)
    ids  = thetas['ids']
    theta = thetas['theta']

    theta_df = pd.DataFrame(data=theta)
    ids_df = pd.DataFrame(data=ids, columns =['INDEX'])

    theta_id = pd.concat([theta_df, ids_df],axis = 1)
    labels = pd.read_pickle(args.label_file)
    labels = labels.drop_duplicates()

    joint = theta_id.merge(labels, on='INDEX', how = 'left')

    full_data = joint.drop(['INDEX','genre'],axis= 1).values

    full_labels = joint['genre'].values   

    data_train,data_test,labels_train,labels_test = train_test_split(full_data, full_labels, test_size=0.2)
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(data_train, labels_train)
    accuracy = str(clf.score(data_test, labels_test))
    print("SCHOLAR ACCURACY: ", accuracy)
    with open(os.path.join(args.output_dir,"accuracy.txt"), 'w') as f:
        f.write(accuracy)
    weights = clf.coef_
    np.savetxt(os.path.join(args.output_dir,"weights.txt"), weights)

if __name__ == '__main__':
    main()
