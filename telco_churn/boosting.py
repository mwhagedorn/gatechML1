import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import learning_curve, GridSearchCV
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn import preprocessing

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

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

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

    plt.show()

def stats_boosting_num_estimators(clf, X_d, y_d, title):

    param_range = np.linspace(1, 30, 20, dtype=np.int32)
    train_scores, test_scores = validation_curve(clf, X_d, y_d, "n_estimators",param_range,cv = 10, n_jobs=-1, verbose=20)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title(title)
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
    plt.show()



from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "Untuned AdaBoost")

# def training_times_per_iteration(clf):
#     print(clf.get_params().keys())
#     params = {'n_iter': [10,20,40, 80, 100]}
#     gs = GridSearchCV(clf, params, scoring='accuracy', cv=10)
#     df = gs.fit(X_train, y_train)
#     print(gs.best_params_)
#
#
# clf = AdaBoostClassifier()
# training_times_per_iteration(clf)

stats_boosting_num_estimators(clf, X_train, y_train, "AdaBoost unscaled number estimators")
# 20
clf = AdaBoostClassifier()
stats_boosting_num_estimators(clf, X_train_std, y_train, "AdaBoost Scaled number estimators")
def stats_boosting_learning_rate(clf, X, y, title):

    param_range = np.linspace(0.01, 0.6, 10)
    train_scores, test_scores = validation_curve(clf, X, y,"learning_rate",param_range,cv = 10, n_jobs=-1, verbose=20)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title(title)
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
    plt.show()

clf = AdaBoostClassifier(n_estimators=20)

stats_boosting_learning_rate(clf, X_train, y_train, "AdaBoost with Unscaled Data")
# lr = 0.4 - this is unscaled data



clf = AdaBoostClassifier(n_estimators=20, learning_rate=0.4)

run_analysis(X_train,y_train, clf, "AdaBoost with 25 estimaters, lrate = 0.4")

clf.fit(X_train, y_train)
print("Feature importances (unscaled):",clf.feature_importances_)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "Tuned AdaBoost")


# scale data
stats_boosting_learning_rate(clf, X_train_std, y_train, "AdaBoost with Scaled Data")


clf = AdaBoostClassifier(n_estimators=20, learning_rate=0.4)
plot_time_complexity(clf, X, y, "AdaBoost Time Complexity")




