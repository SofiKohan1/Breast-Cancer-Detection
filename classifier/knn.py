import numpy as np
import self
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import pandas as pd
from math import sqrt
from collections import Counter
import pprint
from sklearn import metrics

class KNearestNeighbor:

    def __init__(self, k):
        self.k = k

    def get_params(self, deep=True):
        """
        Get available model parameters.
        Required by Scikit-Learn.

        :param deep: Not used.
        :return: Dictionary of parameter names and values.
        """
        return {"k": self.k}
                #"X_test": self.X_test,
                #"X_train": self.X_train}

    def set_params(self, **parameters):
        """
        Set model parameters.
        Required by Scikit-Learn.

        :param parameters: dictionary of parameters and their values.
        :return: self.
        """
        for parameter, value in parameters.item():
            setattr(self, parameters, value)
        return self

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        return self

    def minkowskiDistance(self, test_row, train_row, p=2):
        dist = 0.0
        for i in range(len(test_row)):
            dist += abs(test_row[0]-train_row[0])**p 
       
        return dist**(1/p)

    def predict(self, X_test):
        check_is_fitted(self)
        X = check_array(X_test)
        y_pred = []
        for i in range(len(X)):
            test_row = X[i]
            distances = []
            for j in range(len(self.X_train)):
                train_row = check_array(self.X_train)[j]
                dist = self.minkowskiDistance(test_row, train_row)
                distances.append(dist)
            df_dist = pd.DataFrame(data=distances, columns=['dist'], index=self.y_train)
            sorted_dist = df_dist.sort_values(by=['dist'], axis=0)[:self.k]
            count = Counter(self.y_train[df_dist.index])
            prediction = count.most_common()[0][0]
            y_pred.append(prediction)
        return y_pred

    def score(self, X, y):
        """
        Default scorer (if none provided).
        Required by Scikit-Learn.

        :param X: Feature matrix.
        :param y: Label vector.
        :return: Returns accuracy as the score.
        """
        # print("Scoring...")
        return metrics.accuracy_score(KNearestNeighbor.predict(x_test), y)

