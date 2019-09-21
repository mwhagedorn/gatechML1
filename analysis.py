# Decision trees with some form of pruning
# Neural networks
# Boosting
# Support Vector Machines
# k-nearest neighbors

from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
import pandas as pd

print(LooseVersion(sklearn_version))
#
#
# # from https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier/51398390#51398390
# from sklearn.tree._tree import TREE_LEAF
#
# def is_leaf(inner_tree, index):
#     # Check whether node is leaf node
#     return (inner_tree.children_left[index] == TREE_LEAF and
#             inner_tree.children_right[index] == TREE_LEAF)
#
# def prune_index(inner_tree, decisions, index=0):
#     # Start pruning from the bottom - if we start from the top, we might miss
#     # nodes that become leaves during pruning.
#     # Do not use this directly - use prune_duplicate_leaves instead.
#     if not is_leaf(inner_tree, inner_tree.children_left[index]):
#         prune_index(inner_tree, decisions, inner_tree.children_left[index])
#     if not is_leaf(inner_tree, inner_tree.children_right[index]):
#         prune_index(inner_tree, decisions, inner_tree.children_right[index])
#
#     # Prune children if both children are leaves now and make the same decision:
#     if (is_leaf(inner_tree, inner_tree.children_left[index]) and
#         is_leaf(inner_tree, inner_tree.children_right[index]) and
#         (decisions[index] == decisions[inner_tree.children_left[index]]) and
#         (decisions[index] == decisions[inner_tree.children_right[index]])):
#         # turn node into a leaf by "unlinking" its children
#         inner_tree.children_left[index] = TREE_LEAF
#         inner_tree.children_right[index] = TREE_LEAF
#         ##print("Pruned {}".format(index))
#
# def prune_duplicate_leaves(mdl):
#     # Remove leaves if both
#     decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
#     prune_index(mdl.tree_, decisions

import sklearn
import numpy as np
from sklearn.model_selection import train_test_split

def data_split(examples, labels, train_frac, random_state=None):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
                                        examples, labels, train_size=train_frac, random_state=random_state)

    X_val, X_test, Y_val, Y_test   = train_test_split(
                                        X_tmp, Y_tmp, train_size=0.5, random_state=random_state)

    return X_train, X_val, X_test,  Y_train, Y_val, Y_test



import pandas as pd

data = pd.read_csv('sonar.all-data',header=None)
# https://www.simonwenkel.com/2018/08/23/revisiting_ml_sonar_mines_vs_rocks.html


# identify sonar column names
data.columns = ['X0','X1','X2','X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
                'X10', 'X11','X12','X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19',
                'X20','X21','X22','X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29',
                'X30','X31','X32','X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39',
                'X40','X41','X42','X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49',
                'X50','X51','X52','X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'Class']

data['Class'] = np.where(data['Class'] == 'R',0,1) #Change the Class representation

# shuffle the data rows
data = data.reindex(np.random.permutation(data.index))

X = data.drop('Class',axis=1)
y = data['Class']


print('y actual : \n' +  str(data['Class'].value_counts()))

# make a train, validation, test allocation of 80%, 10%, 10%
X_train, X_val, X_test,  y_train, y_val, y_test = data_split(X, y, 0.8, random_state=23)

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from IPython.display import display
from io import StringIO



# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Unpruned Accuracy:",metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("DT.pdf")


# Create Decision Tree classifer object
clfl2 = DecisionTreeClassifier(max_depth=2, criterion="entropy")

# Train Decision Tree Classifer
clfl2 = clfl2.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clfl2.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy (2 layer):",metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
export_graphviz(clfl2, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("DT2.pdf")

# Create Decision Tree classifer object
clfl3 = DecisionTreeClassifier(max_depth=3, criterion="entropy")

# Train Decision Tree Classifer
clfl3 = clfl3.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clfl3.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy (3 layer):",metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
export_graphviz(clfl3, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("DT3.pdf")

# training curve por favor



# prepruning

# Create Decision Tree classifer object
clf3 = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf3 = clf3.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf3.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Pruned (3) Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Create Decision Tree classifer object
clf5 = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf5 = clf5.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf5.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Pruned (5) Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Create Decision Tree classifer object
clf7 = DecisionTreeClassifier(criterion="entropy", max_depth=7)

# Train Decision Tree Classifer
clf7 = clf7.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf7.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Pruned (5) Accuracy:",metrics.accuracy_score(y_test, y_pred))