import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from pandas import DataFrame
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
data = data.reindex(np.random.RandomState(seed=42).permutation(data.index))


X = data.drop('Class',axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# shuffle the data rows
data = data.reindex(np.random.RandomState(seed=42).permutation(data.index))

def plot_confusion_matrix(y_test,y_pred, title="Confusion Matrix"):

    def get_valid_filename(s):
        s = s.lower()
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)

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
                                                        X_train,
                                                        y_train,
                                                        #_tra Number of folds in cross-validation
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
    plt.savefig('diagrams/boost_lcurve.png')
    plt.show()


def stats_boosting_num_estimators(clf):

    param_range = np.linspace(5, 100, 5, dtype=np.int32)
    train_scores, test_scores = validation_curve(clf, X_train, y_train, "n_estimators",param_range,cv = 10, n_jobs=-1, verbose=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with AdaBoost")
    plt.xlabel("number_of_estimators")
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
    plt.savefig('diagrams/boost_vc_num_estimators.png')
    plt.show()


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()

stats_boosting_num_estimators(clf)
# num_estimators == 25


clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "Untuned AdaBoost CM")


def stats_boosting_learning_rate(clf):

    param_range = np.linspace(0.01, 0.4, 10)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,"learning_rate",param_range,cv = 10, n_jobs=-1, verbose=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with AdaBoost")
    plt.xlabel("learning_rate")
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
    plt.savefig('diagrams/boost_vc_lrate.png')
    plt.show()


clf = AdaBoostClassifier(n_estimators=60, random_state=23)

stats_boosting_learning_rate(clf)

#learning_rate = 0.4


clf = AdaBoostClassifier(n_estimators=25, learning_rate=0.4, random_state=23)


run_analysis(X,y, clf, 'AdaBoost n=25, lr=0.4')

clf = AdaBoostClassifier(n_estimators=25, learning_rate=0.4, random_state=23)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "Tuned AdaBoost CM")

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=60, random_state=23)

run_analysis(X,y, clf, 'GradientBoostingClassifier n=60')

#not better

