from itertools import combinations

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.base import clone

data = pd.read_csv('customer_churn.csv',header=0)


def convert_column(col):
    encoder = preprocessing.LabelEncoder()
    data[col] = encoder.fit_transform(data[col])

data['Churn'] = np.where(data['Churn'] == 'No',0,1) #Change the Class representation
data['gender'] = np.where(data['gender'] == 'Male',0,1)
data['Partner'] = np.where(data['Partner'] == 'No',0,1)
data['PhoneService'] = np.where(data['PhoneService'] == 'No',0,1)
lines_e = preprocessing.LabelEncoder()
data['MultipleLines'] = lines_e.fit_transform(data['MultipleLines'])
is_e = preprocessing.LabelEncoder()
data['InternetService'] = is_e.fit_transform(data['InternetService'])
os_e = preprocessing.LabelEncoder()
data['OnlineSecurity'] = os_e.fit_transform(data['OnlineSecurity'])
convert_column('OnlineBackup')
convert_column('DeviceProtection')
convert_column('TechSupport')
convert_column('StreamingTV')
convert_column('StreamingMovies')
convert_column('Contract')
convert_column('PaperlessBilling')
convert_column('PaymentMethod')
convert_column('Dependents')

data = data.drop('TotalCharges', axis=1)
# shuffle the data rows
data = data.reindex(np.random.RandomState(seed=42).permutation(data.index))
data = data.drop('customerID', axis=1)

# this value is advised to be dropped because it swamps out the other factors
X = data[data.columns[0:-1]]
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

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





def plot_confusion_matrix(y_test,y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test,y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(2.5, 3.5))
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
                                                        random_state=23, verbose=20)

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

    plt.show()


def stats_knn_neighbors(clf, type):
    param_range = np.linspace(30,60, 15, dtype=np.int32)
    train_scores, test_scores = validation_curve(clf, X, y,"n_neighbors",param_range,cv = 10,n_jobs=-1, verbose=20)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with KNN: " + type)
    plt.xlabel("num_neighbors")
    plt.ylabel("Score")
    plt.ylim(0.75, 0.85)
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    max_idx = np.where(test_scores_mean == np.amax(test_scores_mean))
    print(param_range[max_idx])
    plt.show()


from sklearn.neighbors import KNeighborsClassifier


clf = KNeighborsClassifier(weights='uniform')
run_analysis(X_train,y_train, clf, "Default KNN")

clf = KNeighborsClassifier(weights='uniform')
stats_knn_neighbors(clf, 'uniform')
# # k == 47

clf = KNeighborsClassifier(weights='uniform', n_neighbors=47)
clf.fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test.values)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, 'Tuned CM')

# try manhatten distance because all the features differ in what they are
# https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
# Euclidean is a good distance measure to use if the input variables are similar in type (e.g. all measured widths and heights). Manhattan distance is a good measure to use if the input variables are not similar in type (such as age, gender, height, etc.).
clf = KNeighborsClassifier(weights='uniform', n_neighbors=47, p=1)
run_analysis(X, y, KNeighborsClassifier(weights='uniform', n_neighbors=47), "Tuned KNN N=47, manhatten")


run_analysis(X, y, KNeighborsClassifier(weights='uniform', n_neighbors=47), "Tuned KNN N=47")

clf = KNeighborsClassifier(weights='uniform', n_neighbors=47)
plot_time_complexity(clf,X, y, 'KNN Time Complexity')





