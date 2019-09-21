import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

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

unique_elements, counts_elements = np.unique(y, return_counts=True)
print(counts_elements)

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)
dummy = DummyClassifier(random_state=23)
dummy.fit(X_train, y_train)

print("Dummy Accuracy: ", accuracy_score(y_test, dummy.predict(X_test)))
print("Dummy Confusion Matrix:")
print (confusion_matrix(y_test,dummy.predict(X_test)))


y_pred = dummy.predict(X_test)
plot_confusion_matrix(y_test,y_pred,title="Dummy DT")

print("Dummy Precision Score:",precision_score(y_test,y_pred))

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

# clf = DecisionTreeClassifier(random_state=23, criterion="entropy")
# run_analysis(X_train,y_train, clf, title="DT defaults")
# clf = DecisionTreeClassifier(random_state=23, criterion="entropy")
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)


def stats_dt_max_depth(clf):
    param_range = np.linspace(1,25,dtype=np.int32)
    train_scores, test_scores = validation_curve(clf, X_train, y_train,"max_depth",param_range,cv = 10, scoring='accuracy', verbose=50,  n_jobs=-1)

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

# clf = DecisionTreeClassifier(random_state=23, criterion="entropy")
# stats_dt_max_depth(clf)
# clf = DecisionTreeClassifier(random_state=23, criterion="entropy", max_depth=5)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# max_depth = 5

def stats_dt_min_samples_leaf(clf):
    param_range = np.linspace(10,400,100,dtype=np.int32)
    train_scores, test_scores = validation_curve(clf, X_train, y_train ,"min_samples_leaf",param_range,cv = 10,n_jobs=-1, verbose=50)

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

#clf = DecisionTreeClassifier(random_state=23, criterion="entropy")
#stats_dt_min_samples_leaf(clf)
# min samples_leaf = 305

#clf = DecisionTreeClassifier(random_state=23, criterion="entropy", max_depth=5, min_samples_leaf=305)
#clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)


clf = DecisionTreeClassifier(random_state=23, criterion="entropy", max_depth=5, min_samples_leaf=305)

run_analysis(X_train,y_train, clf, "DT - depth 5, min_samples_leaf=300")
clf = DecisionTreeClassifier(random_state=23, criterion="entropy")

run_analysis(X,y, clf, "DT Unprunned")

clf = DecisionTreeClassifier(random_state=23, criterion="entropy", max_depth=5, min_samples_leaf=305)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "Tuned DT")
print("Tuned Accuracy: ", accuracy_score(y_test, y_pred))
print("Tuned Precision Score:",precision_score(y_test,y_pred))