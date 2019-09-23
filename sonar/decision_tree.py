# Import Decision Tree Classifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import re

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

print(__doc__)


data = pd.read_csv('sonar.all-data', header=None)

# identify sonar column names
data.columns = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
                'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19',
                'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29',
                'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39',
                'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49',
                'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'Class']

# Change the Class representation
data['Class'] = np.where(data['Class'] == 'R', 0, 1)

# shuffle the data rows
data = data.reindex(np.random.RandomState(seed=42).permutation(data.index))

X = data.drop('Class', axis=1)
y = data['Class']

def plot_time_complexity(clf, X, y, title='Time curve'):
    """
    Plot the time curve of a classifier
    :param clf: the classifier
    :param X: the entire training set
    :param y: the entire results column
    :param title: the title for the plot
    """
    import time
    training_pct = np.linspace(0.10, 0.9, 10)
    data = []
    for train in training_pct:
        test_pct = 1.0 - train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=23)
        t0 = time.time()
        clf.fit(X_train, y_train)
        t1 = time.time()
        t2 = time.time()
        clf.predict(X_test)
        t3 = time.time()
        times_fit = t1-t0
        times_pred = t3-t2
        data.append([train, times_fit, times_pred])

    data = np.asarray(data)
    train_sizes =data[:,0]
    train_times =data[:,1]
    pred_time = data[:,2]

    # Draw lines
    plt.plot(train_sizes, train_times, '--', color="#111111", label="Training times")
    plt.plot(train_sizes, pred_time, color="#111111", label="Prediction times")

    # Create plot
    plt.title(title)
    plt.xlabel("Training Set Size"), plt.ylabel("Time"), plt.legend(loc="best")
    plt.tight_layout()

    plt.show()

def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(2.5, 3.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

    plt.ylabel('true mine')
    plt.xlabel('\npredicted mine\n\naccuracy={:0.4f}\n precision={:0.4f}'.format(
        accuracy, precision))

    plt.text(0.5, 1.25, title,
             horizontalalignment='center',
             fontsize=12,
             transform=ax.transAxes)

    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

dummy = DummyClassifier(random_state=23)
dummy.fit(X_train, y_train)

print("Dummy Accuracy: ", accuracy_score(y_test, dummy.predict(X_test)))
print("Dummy Confusion Matrix:")
print(confusion_matrix(y_test, dummy.predict(X_test)))

y_pred = dummy.predict(X_test)
plot_confusion_matrix(y_test,y_pred,title="Dummy DT")

print("Dummy Precision Score:", precision_score(y_test, y_pred))


def get_valid_filename(s):
    s = s.lower()
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def run_analysis(X, y, classifier, title):
    train_sizes, train_scores, test_scores = learning_curve(classifier,
                                                            X,
                                                            y,
                                                            # Number of folds in cross-validation
                                                            cv=10,
                                                            # Evaluation metric
                                                            scoring='accuracy',
                                                            # Use all computer cores
                                                            n_jobs=-1,
                                                            # 10 different sizes of the training set
                                                            train_sizes=np.linspace(
                                                                0.10, 1.0, 20),
                                                            shuffle=True,
                                                            random_state=23,
                                                            verbose=True)

    # create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--',
             color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111",
             label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title(title)
    plt.xlabel("Training Set Size"), plt.ylabel(
        "Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("diagrams/" + get_valid_filename(title))
    plt.show()


# baseline DTC LC
clf = DecisionTreeClassifier(random_state=23, criterion="entropy")
run_analysis(X_train, y_train, clf, "Untuned DT Baseline LC")


def stats_dt_max_depth(clf):
    param_range = np.linspace(1, 10, dtype=np.int32)
    train_scores, test_scores = validation_curve(
        clf, X_train, y_train, "max_depth", param_range, cv=10, scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with DT: ")
    plt.xlabel("max_depth")
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


clf = DecisionTreeClassifier(random_state=23, criterion="entropy")
stats_dt_max_depth(clf)
# 8


def stats_dt_min_samples_leaf(clf):
    param_range = np.linspace(10, 120, 10, dtype=np.int32)
    train_scores, test_scores = validation_curve(
        clf, X_train, y_train, "min_samples_leaf", param_range, cv=10,)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with DT: ")
    plt.xlabel("min_samples_leaf")
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


clf = DecisionTreeClassifier(random_state=23, criterion="entropy")
stats_dt_min_samples_leaf(clf)
# 30
run_analysis(X,y, DecisionTreeClassifier(random_state=23, min_samples_leaf=30, max_depth=4), "Tuned DT depth=4, msl=30")


clf = DecisionTreeClassifier(random_state=23, min_samples_leaf=30, max_depth=4)

clf.fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test.values)
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred, title="DT depth=4, min_samp_leaf=30 ")

clf = DecisionTreeClassifier(random_state=23, min_samples_leaf=30, max_depth=4)
plot_time_complexity(clf, X, y, "DT Time Complexity")

