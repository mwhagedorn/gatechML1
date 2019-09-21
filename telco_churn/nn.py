import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import learning_curve
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

X = data[data.columns[0:-1]]
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

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


from sklearn.neural_network import MLPClassifier

# see https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# why 32? input_size + output_size/2  61/2


def stats_nn_with_hidden_layer(clf):

    param_range = [(1,), (2,), (4,),(6,) ,(8,), (10,), (12,), (14,), (16,), (20,), (32,)]
    x_axes = [val[0] for val in param_range]
    train_scores, test_scores = validation_curve(clf, X_train, y_train, "hidden_layer_sizes",param_range,cv = 10, n_jobs=-1, verbose=50)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with NN, single hidden layer")
    plt.xlabel("hidden_layer_size")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.plot(x_axes, train_scores_mean, label="Training score", color="r")
    plt.fill_between(x_axes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(x_axes, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(x_axes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()

def loss_curve(X,y, classifier, title):
    classifier.fit(X,y)
    plt.plot(classifier.loss_curve_)
    plt.title(title)
    plt.xlabel("Number of Steps"), plt.ylabel("Loss function")
    plt.show()



clf = MLPClassifier()
loss_curve(X_train, y_train, clf, "Loss Curve, Rate=0.001")

clf = MLPClassifier(solver='lbfgs',
                    random_state=23,
                    shuffle=True,
                    activation='relu'
                    )

run_analysis(X_train, y_train, clf, "NN with default learning rate")


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "Untuned NN")

clf = MLPClassifier(solver='lbfgs',
                    random_state=23,
                    shuffle=True,
                    activation='relu'
                    )
stats_nn_with_hidden_layer(clf)
# 10 seems best

def stats_nn_with_max_iteration(clf):
    param_range = np.linspace(1, 15, 10)
    train_scores, test_scores = validation_curve(clf, X_train, y_train, "max_iter", param_range, cv=10)


    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with NN, single hidden layer (16 units)")
    plt.xlabel("epochs")
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

clf = MLPClassifier(solver='lbfgs',
                    random_state=23,
                    shuffle=True,
                    activation='relu',
                    hidden_layer_sizes=(10,)
                    )

stats_nn_with_max_iteration(clf)


# use 10 epochs

clf = MLPClassifier(solver='lbfgs',
                    random_state=23,
                    shuffle=True,
                    activation='relu',
                    hidden_layer_sizes=(16,),
                    max_iter=10
                    )


run_analysis(X,y, clf, "NN 1 Layer, 16 units, max_iter=10")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "Tuned NN")