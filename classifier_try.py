__author__ = 'manshu'

import csv
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import sys
import pylab as pl

num_lines = 10000

file_name = "a.csv"
lines = []
data_class = []
with open(file_name, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        data_class.append(row[1])
        lines.append(row[2:])

features = np.array(lines[0])

data_class = data_class[1:]
data = lines[1:]
print len(features)
print len(data_class)
print len(data)
print features
print data[0]

clf = RandomForestClassifier()#compute_importances=True
clf.fit(data, data_class)

# from the calculated importances, order them from most to least important
# and make a barplot so we can visualize what is/isn't important
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(len(features)) + 0.5
pl.barh(padding, importances[sorted_idx], align='center')
pl.yticks(padding, features[sorted_idx])
pl.xlabel("Relative Importance")
pl.title("Variable Importance")
pl.show()

sys.exit(0)

dt = tree.DecisionTreeClassifier()
clf = dt.fit(data, data_class)

def get_lineage(tree, feature_names):
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]

     def recurse(left, right, child, lineage=None):
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     for child in idx:
          for node in recurse(left, right, child):
               print node

#get_lineage(clf, features)
from StringIO import StringIO
out = StringIO()
out = tree.export_graphviz(clf, out_file='car_tree.dot', feature_names=features)
import os
os.system("dot -Tpng car_tree.dot -o car_tree.png")