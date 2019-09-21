import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
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
# shuffle the data rows
data = data.reindex(np.random.RandomState(seed=42).permutation(data.index))

X = data.drop('Class',axis=1)
y = data['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

def get_valid_filename(s):
    s = s.lower()
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

def plot_confusion_matrix(y_test,y_pred, title="Confusion Matrix"):


    cm = confusion_matrix(y_test,y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(2.6, 3.6))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s = cm[i, j], va = 'center', ha = 'center')

    plt.ylabel('true churn')
    plt.xlabel('\npredicted churn\n\naccuracy={:0.4f}\n precision={:0.4f}'.format(accuracy, precision))

    plt.text(0.5, 1.25, title,
             horizontalalignment='center',
             fontsize=12,
             transform=ax.transAxes)

    plt.savefig("diagrams/"+get_valid_filename(title))
    plt.show()




def run_analysis(X,y, classifier, title):
    train_sizes, train_scores, test_scores = learning_curve( classifier,
                                                        X,
                                                        y,
                                                        # Number of folds in cross-validation
                                                        cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1,
                                                        # 10 different sizes of the training set
                                                        train_sizes=np.linspace(0.10,1.0, 20),
                                                        shuffle=True,
                                                        random_state=23)

    #create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title(title)
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
   # plt.savefig("diagrams/" + get_valid_filename(title))
    plt.show()

from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# kernel
# regularization
# gamma

def stats_svc_gamma(clf, title):

    param_range = np.linspace(0.0001, 2, 10, dtype=np.int32)
    train_scores, test_scores = validation_curve(clf, X_train, y_train, "gamma",param_range,cv = 10, n_jobs=-1, verbose=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title(title)
    plt.xlabel("gamma")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()

def stats_svc_C(clf, type='rbf'):

    param_range = np.linspace(0.1,5, 100)
    train_scores, test_scores = validation_curve(clf, X_train, y_train, "C",param_range,cv = 10, n_jobs=-1, verbose=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with SVM: "+type )
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()



clf = svm.SVC(kernel='rbf', C=750, random_state=23)
#stats_svc_gamma(clf)
# # gamma is 1.0
# #
clf = svm.SVC(kernel='rbf', random_state=23)
stats_svc_C(clf)
#C is 2

def C_range(c):
    clf = svm.SVC(kernel='rbf', C=c, gamma=1, random_state=23)
    run_analysis(X_train, y_train, clf, 'SVM(rbf) gamma=1, c='+str(c))

# winner winner chicken dinner
#C_range(2)

clf = svm.SVC(kernel='rbf', C=2, random_state=23, gamma=1.0 )
#run_analysis(X,y, clf, "SVC(rbf) C=2, gamma=1")
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#print (confusion_matrix(y_test,y_pred))
#plot_confusion_matrix(y_test, y_pred, "Tuned SVC (rbf)")


clf = svm.SVC(kernel='linear', random_state=23, C=1.5 )
stats_svc_C(clf, type='linear')

clf = svm.SVC(kernel='linear', random_state=23, C=1.1 )
run_analysis(X_train, y_train, clf, 'SVM(linear)')

# clf = svm.SVC(kernel='linear', random_state=23)
# stats_svc_C(clf, type='linear')
# #1.25 for linear
clf = svm.SVC(kernel='linear', C=1, random_state=23)

# run_analysis(X,y, clfl, 'SVM(linear)')
#
clf = svm.SVC(kernel='linear', C=1.25, random_state=23)
clf.fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test.values)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, title="SVC linear, C one-two-five")
#
# clf = svm.SVC(kernel='linear', random_state=23)
# clf.fit(X_train.values, y_train.values)
# y_pred = clf.predict(X_test.values)
# print (confusion_matrix(y_test,y_pred))
# plot_confusion_matrix(y_test, y_pred, title="SVC linear defaults")
#
#
# clf = svm.SVC(kernel='rbf', C=1.25, random_state=23)
# run_analysis(X,y, clf, 'SVM(rbf) C_one-two-five')
#
# clf = svm.SVC(kernel='rbf', C=8.0, random_state=23)
# run_analysis(X,y, clf, 'SVM(rbf) C_eight')
#
# clf = svm.SVC(kernel='rbf', C=10, random_state=23)
# run_analysis(X,y, clf, 'SVM(rbf) C_ten')
#
# clf = svm.SVC(kernel='rbf', C=15, random_state=23)
# run_analysis(X,y, clf, 'SVM(rbf) C_fifteen')




# the C parameter trades off correct classification of training examples against maximization of the decision
# function’s margin. For larger values of C, a smaller margin will be accepted if the decision function is better at
# classifying all training points correctly. A lower C will encourage a larger margin, therefore a simpler decision
# function, at the cost of training accuracy. In other words``C`` behaves as a regularization parameter in the SVM.

# try scaling
# clf = make_pipeline(StandardScaler(),clfl)
# run_analysis(X,y, clf, 'SVM(linear) - scaled data')





#NOTE:  linear svm tend to do better on high dimensional spaces
#https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf

# # this model doesnt benefit from higher degrees poly, too high bias low accuracy
# clf = svm.SVC(kernel='poly', degree=3, random_state=23)
# run_analysis(X,y, clf, 'SVM(poly-3)')
#
# clf = svm.SVC(kernel='poly', degree=2, random_state=23)
# run_analysis(X,y, clf, 'SVM(poly-2)')
#
# clf = svm.SVC(kernel='rbf', gamma=1.0, random_state=23, C=0.75)
# run_analysis(X,y, clf, 'SVM(rbf, 1,0.75)')
# C is 0.75




