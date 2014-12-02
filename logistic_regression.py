__author__ = 'manshu'


import pandas as pd
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
import random
import sys
import csv

# numerical_cols = ['VehOdo', 'VehicleAge', 'VehBCost', 'WarrantyCost', 'MMRCurrentAuctionCleanPrice',
#                   'MMRCurrentRetailAveragePrice', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
#                   'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentRetailCleanPrice']

numerical_cols = ['VehOdo',  'VehicleAge', 'VehBCost', 'WarrantyCost',
                  'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
                  'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice',
                  'ProfitAcquisitionAverage', 'ProfitAcquisitionClean', 'ProfitCurrentAverage', 'ProfitCurrentClean', 'AverageProfit', 'CleanProfit', 'OdoPAge']

logistic_fx = lambda x: 1.0 / (1.0 + math.e**(-1 * x))
learning_rate = 0.015
max_iter = 250

def readFile(file_name):
    f = open(file_name, 'r')
    training_data = []
    training_class = []
    for line in f:
        if line == "":
            continue
        line = line.strip()
        row_data = line.split(",")
        training_data.append([float(row_data[0]), float(row_data[1])])
        training_class.append(int(row_data[2]))
    return training_data, training_class

def sigmoid_fx(linear_hypothesis):
    M, N = np.shape(linear_hypothesis)

    sigmoid_hypothesis = np.zeros([M, N])

    for i in range(0, M):
        for j in range(0, N):
            sigmoid_hypothesis[i, j] = logistic_fx(linear_hypothesis[i, j])

    return sigmoid_hypothesis

def predict(X, theta):
    hypothesis = sigmoid_fx(X.dot(theta))
    predictions = np.zeros(hypothesis.shape)

    for i in range(0, hypothesis.size):
        if hypothesis[i] >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return hypothesis, predictions

def costFunction(theta, X, Y):
    M = Y.size
    Jtheta = 0

    gradient = np.zeros(np.shape(theta))

    linear_hypothesis = X.dot(theta)
    hypothesis = sigmoid_fx(linear_hypothesis)

    Jtheta = np.sum((-1 * Y).T.dot(np.log(hypothesis)) - (1 - Y).T.dot(np.log(1 - hypothesis))) / (1.0 * M)

    error = (hypothesis - Y)

    for j in range(0, gradient.size):
        gradient[j] = np.sum(error.T.dot(X[:, j])) / (1.0 * M)

    return Jtheta, gradient

def normalize_data(X, features):
    M, N = np.shape(X)
    for j in range(1, N):
        # match = False
        # for ftr in numerical_cols:
        #     if features[j-1].startswith(ftr):
        #         match = True
        # if not match:
        #     continue
        # if features[j-1] not in numerical_cols:
        #     continue
        mean = np.mean(X[:, j])
        #mm = np.max(X[:, j]) - np.min(X[:, j])
        mm = np.std(X[:, j])
        X[:, j] = (X[:, j] - mean) / (1.0 * mm)

    return X

def preprocess(X, features):
    M, N = np.shape(X)


    X = np.append(np.ones([M, 1]), X, 1)
    X = normalize_data(X, features)

    return X

def getAccuracy(predictions, Y):

    M = Y.size
    sum_val = 0
    for i in range(0, M):
        if predictions[i] == Y[i]:
            sum_val += 1
    accuracy = sum_val / (1.0 * M)
    return accuracy

def logistic_regression(X, Y, features):

    #X, Y = readFile("lr_t1.txt")

    X = np.array(X)
    Y = np.matrix(Y).T

    M, N = np.shape(X)

    # positives = list(np.where(Y == 1))[0]
    # negatives = list(np.where(Y == 0))[0]
    #
    # s1 = [200 for i in range(0, len(positives))]
    # s2 = [100 for i in range(0, len(negatives))]

    # plt.scatter(X[positives, 0], X[positives, 1], s=s1, marker='+')
    #
    # plt.scatter(X[negatives, 0], X[negatives, 1], s=s2, marker='o')
    #
    # plt.xlabel('Exam 1 Score')
    # plt.ylabel('Exam 2 Score')
    #
    theta_param = np.zeros([N + 1, 1])
    for i in range(0, N + 1):
        theta_param[i] = random.random()

    theta_param = np.array([[ 0.79614327], [ 0.64151511], [ 0.76398154], [ 0.36273963], [ 0.32608115], [ 0.37254119], [ 0.03952807], [ 0.10979644], [ 0.82724346], [ 0.3982566 ], [ 0.75496772], [ 0.20258409], [ 0.35894928], [ 0.33887323], [ 0.65930873], [ 0.48777835], [ 0.89330239], [ 0.80053253], [ 0.34298245], [ 0.20129628]])

    # theta_param = np.array([[ 0.67605752], [ 0.55168346], [ 0.27144959], [ 0.73673547], [ 0.15480109], [ 0.97850866], [ 0.44643256], [ 0.68159934], [ 0.89509255], [ 0.98479877],
    #                [ 0.4428826 ], [ 0.76526354], [ 0.42932508], [ 0.6934041 ], [ 0.34877231], [ 0.03783622], [ 0.19395029], [ 0.80813425], [ 0.27508955], [ 0.10439405]]) #0.05031 #0.02 100
    #theta_param = np.array([[0.49334852], [0.44310159],  [0.62672332],  [0.42342171],  [0.54510326]]) #0.04775 # 0.02 # 100 #normalization on
    print "Theta = ", theta_param

    X = preprocess(X, features)

    cost, grad = costFunction(theta_param, X, Y)
    print cost
    print grad
    #plt.show()

    #X = normalize_data(X)

    # for j in range(1, N + 1):
    #     mean = np.mean(X[:, j])
    #     mm = np.max(X[:, j]) - np.min(X[:, j])
    #     X[:, j] = (X[:, j] - mean) / (1.0 * mm)

    prev_J_theta = 0.0

    for run_id in range(0, max_iter):
        # for trId in range(0, M):
        #     for attr_id in range(0, N + 1):
        #         tuple_value[attr_id] = X[trId, attr_id]
        #
        #     #print tuple_value
        #     px = logistic_fx(theta_param.dot(tuple_value))
        #     predictions[trId] = px
        hypothesis, predictions = predict(X, theta_param)

        prev_J_theta, gradient = costFunction(theta_param, X, Y)

        error = hypothesis - Y
        for theta_id in range(0, N + 1):
            theta_param[theta_id] -= learning_rate * gradient[theta_id] # * (np.sum(error.T.dot(X[:, theta_id])) / (1.0 * M))

        Jtheta, gradient = costFunction(theta_param, X, Y)

        if abs(abs(prev_J_theta) - abs(Jtheta)) < 10 ** -3:
            print "Breaked at ", run_id
            print Jtheta
            break
        #prev_J_theta = Jtheta

        # print Jtheta
        # print theta_param

    print prev_J_theta
    print theta_param

    hypothesis, predictions = predict(X, theta_param)

    accuracy = getAccuracy(predictions, Y)
    print "Accuracy = ", accuracy

    return theta_param, accuracy
    # X1, Y1 = readFile("lr_t2.txt")
    # X1 = np.array(X1)
    # Y1 = np.matrix(Y1).T
    #
    # M1, N1 = np.shape(X1)
    # X1 = np.append(np.ones([M1, 1]), X1, 1)
    # # for j in range(1, N1 + 1):
    # #     mean = np.mean(X1[:, j])
    # #     mm = np.max(X1[:, j]) - np.min(X1[:, j])
    # #     X1[:, j] = (X1[:, j] - mean) / (1.0 * mm)
    #
    # hypothesis, predictions = predict(X1, theta_param)
    #
    # sum_val = 0
    # for i in range(0, M1):
    #     if predictions[i] == Y1[i]:
    #         sum_val += 1
    # print "Accuracy = ", sum_val / (1.0 * M1)

def read_csv_file(file):
   with open(file, 'rb') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    line = 0
    features = []
    class_labels = []
    classifier_data = []

    for row in csv_reader:
        line += 1
        if line == 1:
            features = row[1:]
            continue
        class_labels.append(int(row[0]))
        attribute_vals = []
        attrs = row[1:]
        for col_id in range(0, len(attrs)):
            attr_val = attrs[col_id]
            if features[col_id] in numerical_cols:
                attribute_vals.append(int(math.ceil(float(attr_val))))
        classifier_data.append(attribute_vals)

    return classifier_data, class_labels, features

if __name__ == "__main__":
    X, Y, features = read_csv_file("x.csv")#("training_equal_freq.csv")
    M, N = np.shape(X)

    num_training = int(0.6 * M)
    print num_training
    X_training = X[:num_training]
    Y_training = Y[:num_training]

    X_CV = X[num_training:]
    Y_CV = Y[num_training:]

    X_CV = np.array(X_CV)
    X_CV = preprocess(X_CV, features)
    Y_CV = np.matrix(Y_CV).T

    accuracy_CV = 0.0

    max_theta_params = []
    max_accuracy = []
    max_run = 5
    run_id = 0

    while run_id < max_run:
        theta_params, accuracy_tr = logistic_regression(X_training, Y_training, features)

        hypothesis, predictions = predict(X_CV, theta_params)
        accuracy_CV = getAccuracy(predictions, Y_CV)

        if accuracy_CV >= 0.85:
            break
        print "N->Cross Validation accuracy = ", accuracy_CV

        max_theta_params.append(theta_params)
        max_accuracy.append(accuracy_CV)

        run_id += 1

    if run_id == max_run:
        accuracy_CV = max(max_accuracy)
        accuracy_id = max_accuracy.index(accuracy_CV)
        theta_params = max_theta_params[accuracy_id]

    print "Cross Validation accuracy = ", accuracy_CV
    X1, ref_ids, features1 = read_csv_file("y.csv")#("test_equal_freq.csv")

    X1 = np.array(X1)

    X1 = preprocess(X1, features1)
    hypothesis, predictions = predict(X1, theta_params)

    output = []
    #output.append(['RefId', 'IsBadBuy'])
    for i in range(0, len(ref_ids)):
        output.append([int(ref_ids[i]), int(predictions[i])])

    R = np.array(output)

    out_file = np.asarray(R)

    np.savetxt("result.csv", out_file, delimiter=",", fmt="%s")

