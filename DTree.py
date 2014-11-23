__author__ = 'manshu'

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

# dummy data:
df = pd.DataFrame({'col1':[0,1,2,3],'col2':[3,4,5,6],'dv':[0,1,0,1]})

# create decision tree
dt = tree.DecisionTreeClassifier()
iris = load_iris()
#clf = dt.fit(df.ix[:,:2], df.dv)
clf = dt.fit(iris.data, iris.target)

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
#get_lineage(dt, df.columns)
get_lineage(clf, iris.feature_names)
from StringIO import StringIO
out = StringIO()
out = tree.export_graphviz(clf, out_file='tree.dot', feature_names=iris.feature_names)
print iris.data
print iris.target_names
print iris.target
print iris.keys()
import os
os.system("dot -Tpng tree.dot -o tree.png")


