
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def plot_time_complexity(clf, X, y, title='Time curve'):
    """
    Plot the time curve of a classifier
    :param clf: the classifier
    :param X: the entire training set
    :param y: the entire results column
    :param title: the title for the plot
    """
    import timeit
    TEST_CODE_FIT = '''
clf.fit(X_train, y_train)'''
    TEST_CODE_PRED = '''
clf.predict(X_test)'''

    training_pct = np.linspace(0.10, 1.0, 10)
    df = pd.DataFrame(columns=['train_pct', 'train_time', 'pred_time'])
    for train in training_pct:
        test_pct = 1.0 - train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=23)
        times_fit = timeit.timeit(setup='pass', stmt=TEST_CODE_FIT, number=2)
        times_fit = times_fit / 2.0
        times_pred = timeit.timeit(setup='pass', stmt=TEST_CODE_PRED, number=2)
        times_pred = times_pred / 2.0
        df.append(np.array([train, times_fit, times_pred]))

    train_sizes = df['train_pct'].to_numpy()
    train_times = df['train_time'].to_numpy()
    pred_time = df['pred_time'].to_numpy()

    # Draw lines
    plt.plot(train_sizes, train_times, '--', color="#111111", label="Training times")
    plt.plot(train_sizes, pred_time, color="#111111", label="Cross-validation score")

    # Create plot
    plt.title(title)
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()

    plt.show()
