__author__ = 'manshu'

from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import random


if __name__=="__main__":
    training_data = [[1., 0.5, 1],
                    [2., 1.2, 1],
                    [2.5, 2., 1],
                    [3., 2., 1],
                    [1.5, 2., -1],
                    [2.3, 3., -1],
                    [1.2, 1.9, -1],
                    [0.8, 1., -1],
                    ]

    test_data = [[2.7, 2.7, 1],
                [2.5, 1., 1],
                [1.5, 2.5, -1],
                [1.2, 1., -1],
                ]

    training_data = np.matrix(training_data)
    test_data = np.matrix(test_data)

    # print training_data[:, 0]
    # print training_data[:, 1]
    # print training_data[:, 2]
    positives = list(np.where(training_data[:, 2] == 1.))[0]
    negatives = list(np.where(training_data[:, 2] == -1.))[0]
    #
    # print positives
    # print negatives

    s1 = [200 for i in range(0, len(positives))]
    s2 = [100 for i in range(0, len(negatives))]


    # #configure  X axes
    # plt.xlim(0.,4.)
    # plt.xticks(np.arange(0., 4.5, 0.5))
    #
    # plt.ylim(0.,4.)
    # plt.yticks(np.arange(0., 4.5, 0.5))
    #
    # plt.plot(training_data[positives, 0], training_data[positives, 1], marker='o', color='red', ms=20)
    #
    # plt.plot(training_data[negatives, 0], training_data[negatives, 1], marker='x', color='blue', ms=20)
    #
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.show()

    M, N = np.shape(training_data)
    M1, N1 = np.shape(test_data)
    if N != N1:
        print "Size not same"
        sys.exit(0)

    K = 3
    test_class = []
    for ti in range(0, M1):
        distances = []
        print "For point in test = ", test_data[ti, 0:2]
        for tri in range(0, M):
            diff = 0.
            for ai in range(0, N - 1):
                diff += (test_data[ti, ai] - training_data[tri, ai]) ** 2
            diff = math.sqrt(diff)
            distances.append([diff, training_data[tri, N-1]])

            print "For point in training ", training_data[tri, 0:2]
            print "Distance = ", diff

        print "Unsorted distances = ", distances
        distances.sort(key=lambda x: x[0])
        print "Sorted Distances = ", distances
        class_count = {}
        for tuple in distances[0:K]:
            if tuple[1] not in class_count:
                class_count[tuple[1]] = 0
            class_count[tuple[1]] += 1
        print "Class Count = ", class_count
        max_value = max(class_count.values())
        print "Max Value = ", max_value
        max_class = []
        for tuple_class, distance in class_count.items():
            if distance == max_value:
                max_class.append(tuple_class)

        if len(max_class) == 1:
            test_class.append(max_class[0])
        else:
            print "Multiple class match. Picking random for ", ti
            chosen_class = random.choice(max_class)
            print "Chosen class = ", chosen_class
            test_class.append(chosen_class)

    match = 0
    for i in range(0, M1):
        print test_data[i, 2], test_class[i]
        if test_class[i] == test_data[i, 2]:
            match += 1

    print "Match = ", match
    print "Accuracy = ", match / (1.0 * M1)