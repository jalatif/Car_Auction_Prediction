__author__ = 'manshu'

import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from pandas import DataFrame

from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.lda import LDA

def predict(training_file, test_file):

    line = 0
    features = []
    class_labels = []
    classifier_data = []
    test_data = []
    with open(training_file, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in csv_reader:
            line += 1
            if line == 1:
                features = row[2:]
                continue
            class_labels.append(int(row[1]))
            attribute_vals = []
            for attr_val in row[2:]:
                attribute_vals.append(int(attr_val))
            classifier_data.append(attribute_vals)

    print features
    print class_labels

    X = np.array(classifier_data)
    Y = np.array(class_labels)

    clf1 = GaussianNB()
    clf1.fit(X, Y)
    y_pred1 = clf1.predict(X)

    print y_pred1

    print (y_pred1 == Y).sum() / (1.0 * len(classifier_data))

    clf2 = DecisionTreeClassifier(random_state=0)
    clf2.fit(X, Y)
    y_pred2 = clf2.predict(X)

    print y_pred2

    print (y_pred2 == Y).sum() / (1.0 * len(classifier_data))

    clf3 = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA(n_components=2)),
    ('classifier', GaussianNB())
    ])

    clf3.fit(X, Y)      # fitting on the training dataset
    y_pred3 = clf3.predict(X) # classifying the test dataset

    print y_pred3

    print (y_pred3 == Y).sum() / (1.0 * len(classifier_data))

    ref_ids = []

    with open(test_file, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        line = 0
        for row in csv_reader:
            line += 1
            if line == 1:
                features = row[1:]
                continue
            ref_ids.append(int(row[0]))

            attribute_vals = []
            for attr_val in row[1:]:
                attribute_vals.append(int(attr_val))
            test_data.append(attribute_vals)

    print features

    Z = np.array(test_data)
    #R = np.array(ref_ids)
# R = np.array(output)
    #
    # out_file = np.asarray(R)
    #
    # np.savetxt("result.csv", out_file, delimiter=",")

    y_pred11 = clf1.predict(Z)

    #print y_pred3

    y_pred22 = clf2.predict(Z)

    y_pred33 = clf3.predict(Z)

    # R = np.array(output)
    #
    # out_file = np.asarray(R)
    #
    # np.savetxt("result.csv", out_file, delimiter=",")

    print y_pred33
    print (y_pred33 == y_pred11).sum() / (1.0 * len(test_data))
    print ref_ids

    #N = np.array(ref_ids + test_data)
    output = []
    #output.append(['RefId', 'IsBadBuy'])
    for i in range(0, len(ref_ids)):
        output.append([ref_ids[i], y_pred33[i]])

    print output
    R = np.array(output)

    out_file = np.asarray(R)

    np.savetxt("result.csv", out_file, delimiter=",", fmt="%s")

    #print cross_val_score(clf2, X, Y, cv=10)


    # score = cross_val_score(estimator, classifier_data, class_data).mean()
    # print score



if __name__=="__main__":
    training_file = "x.csv"
    test_file = "y.csv"
    predict(training_file, test_file)
