import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
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

def loss_curve(X,y, classifier, title):
    classifier.fit(X,y)
    plt.plot(classifier.loss_curve_)
    plt.title(title)
    plt.xlabel("Number of Steps"), plt.ylabel("Loss function")
    plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier()
#loss_curve(X_train, y_train, clf, "Loss Curve, Rate=0.001")

clf = MLPClassifier(learning_rate_init=0.29)
#loss_curve(X_train, y_train, clf, "Loss Curve, Rate=0.29")

clf = MLPClassifier(learning_rate_init=0.1)
#run_analysis(X_train, y_train, clf, "MLP learning rate 0.1")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "MLP all defaults")

# see https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# why 32? input_size + output_size/2  61/2


def stats_nn_with_hidden_layer():

    clf = MLPClassifier(solver='sgd',
                    random_state=23,
                    max_iter=2000,
                    shuffle=True,
                    learning_rate_init=0.1,
                    activation='relu')


    param_range = [(4,),(6,),(8,),(10,),(14,) ,(16,), (32,), (48,), (64,), (74, ), (84,)]
    x_axes = [val[0] for val in param_range]
    train_scores, test_scores = validation_curve(clf, X_train, y_train, "hidden_layer_sizes",param_range,cv = 10, verbose=True, n_jobs=-1)

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




# stats_nn_with_hidden_layer()
#
# # pick 70
#
# # how is the tuned NN doing
clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu', learning_rate_init=0.15)
run_analysis(X_train, y_train,clf, "NN with lrate=0.15, 70 units in hidden layer")

clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu', learning_rate_init=0.15, alpha=0.45)
run_analysis(X_train, y_train,clf, "NN with lrate=0.15, 70 units in hidden layer, alpha 0.45")

clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu', learning_rate_init=0.15, alpha=0.45)
clf.fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test.values)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, title="NN (70,) lrate=0.15, alpha 0.45")

def stats_nn_with_max_iteration():
    clf = MLPClassifier(solver='lbfgs',
                    hidden_layer_sizes=(34,),
                    random_state=23,
                    shuffle=True,
                    activation='relu'
                )

    param_range = np.linspace(5, 70, 5)
    train_scores, test_scores = validation_curve(clf, X, y, "max_iter", param_range, cv=10)


    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with NN, single hidden layer")
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


stats_nn_with_max_iteration()

clf = MLPClassifier(solver='lbfgs',
                    hidden_layer_sizes=(34,),
                    max_iter=55,
                    random_state=23,
                    shuffle=True,
                    activation='relu'
                    )

run_analysis(X,y, clf, 'Neural Net, 1 layer, 34 units, 55 Epochs')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "Tuned NN")

clf = MLPClassifier(solver='lbfgs',
                    random_state=23,
                    shuffle=True,
                    activation='relu'
                    )

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print (confusion_matrix(y_test,y_pred))
plot_confusion_matrix(y_test, y_pred, "Untuned NN")


#
#
#
#
# # Notes:
# # https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
# # https://www.mikulskibartosz.name/precision-vs-recall-explanation/
# # optimization and rules of thumb: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# # https://s3.amazonaws.com/heatonresearch-books/free/Encog3Java-User.pdf
#
# from sklearn.model_selection import validation_curve
#
clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(70,), random_state=23, shuffle=True, activation='relu', learning_rate_init=0.15, alpha=0.45, verbose=20)
plot_time_complexity(clf, X, y, "NN Time complexity")