__author__ = 'manshu'

from matplotlib import pyplot as plt
import numpy as np
import math
import sys

def dist(p1, p2):
    l1 = p1.size
    l2 = p2.size
    if l1 != l2:
        return -1
    distance = 0.
    for attr_id in range(0, l1):
        distance += (p1[0, attr_id] - p2[0, attr_id]) ** 2

    return math.sqrt(distance)

def kmeans(data, K, initial_clusters=[]):

    data = np.matrix(data)
    M, N = np.shape(data)

    if initial_clusters == []:
        pass #choose random

    print "Initial Cluster = ", initial_clusters

    current_clusters = np.matrix(initial_clusters)

    current_id = 0
    assigned_clusters = []

    while True:#current_id < 2:
        plt.figure(current_id)
        #configure  X axes
        plt.xlim(0, 9)
        plt.xticks(np.arange(0, 9, 1))

        plt.ylim(0, 9)
        plt.yticks(np.arange(0, 9, 1))
        plt.xlabel('X')
        plt.ylabel('Y')

        colors = ['red', 'green', 'blue', 'yellow', 'pink', 'brown', 'black']

        assigned_clusters = []

        for object_id in range(0, M):
            object = data[object_id, :]
            ci_dist = []
            for ci_d in range(0, K):
                ci = current_clusters[ci_d, :]
                ci_dist.append(dist(object, ci))
            min_dist = min(ci_dist)
            assigned_clusters.append(ci_dist.index(min_dist))

        #print assigned_clusters

        new_clusters = np.matrix(current_clusters)

        cluster_same = True

        for cluster_id in range(0, K):
            my_objects = [i for i in range(0, len(assigned_clusters)) if assigned_clusters[i] == cluster_id]
            plt.plot(data[my_objects, 0], data[my_objects, 1], ls='--', marker='x', color=colors[cluster_id], ms=20)
            plt.plot(data[[my_objects[0], my_objects[-1]], 0], data[[my_objects[0], my_objects[-1]], 1], ls='--', marker='x', color=colors[cluster_id], ms=20) #For connecting end points
            plt.plot(current_clusters[cluster_id, 0], current_clusters[cluster_id, 1], marker='o', color=colors[cluster_id], ms=10)


            for attr_id in range(0, N):
                mean_attr = 0.
                for object_id in my_objects:
                    mean_attr += data[object_id, attr_id]
                mean_attr /= (1. * len(my_objects))
                new_clusters[cluster_id, attr_id] = mean_attr
                if mean_attr != current_clusters[cluster_id, attr_id]:
                    cluster_same = False

        #print current_clusters
        print "Cluster at round ", current_id, " = ", new_clusters

        #print cluster_same

        current_clusters = new_clusters

        if cluster_same:
            break
        current_id += 1

    plt.show()

    cluster_variance = 0

    for cluster_id in range(0, K):
        ci = current_clusters[cluster_id, :]
        my_objects = [i for i in range(0, len(assigned_clusters)) if assigned_clusters[i] == cluster_id]
        for object_id in my_objects:
            object = data[object_id, :]
            cluster_variance += dist(object, ci) ** 2

    return cluster_variance

if __name__=="__main__":

    data = [
        [1., 1.],
        [1.5, 2.],
        [2.5, 5.5],
        [6., 5.],
        [4., 5.],
        [4.5, 5.],
        [3.5, 4.5],
        [5., 1.],
        [6., 1.],
    ]
    k = 3
    initial_clusters1 = [
        [4., 5.],
        [3.5, 4.5],
        [6., 5.]
    ]

    initial_clusters2 = [
        [2.5, 5.5],
        [3.5, 4.5],
        [6., 1.]
    ]

    cluster_variance = kmeans(data, 3, initial_clusters1)

    print cluster_variance
