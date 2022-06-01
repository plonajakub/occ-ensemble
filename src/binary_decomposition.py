import numpy as np
from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class BinaryDecompositionEnsemble(BaseEnsemble):
    def __init__(self,
                 base_classifier=KNeighborsClassifier(n_neighbors=10),
                 random_state=None,):
        self.base_classifier = base_classifier
        self.classifiers = None
        self.random_state = random_state
        self.classes = None
        self.n_features = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        self.classifiers = {c1: {} for c1 in self.classes}

        X_y_combined = np.zeros((X.shape[0], X.shape[1] + 1))
        X_y_combined[:, :-1] = X
        X_y_combined[:, -1] = y

        for c1 in self.classes:
            for c2 in self.classes:
                if c1 >= c2:
                    continue
                X_y_c1 = X_y_combined[X_y_combined[:, -1] == c1]
                X_y_c2 = X_y_combined[X_y_combined[:, -1] == c2]
                X_y_c1_c2 = np.concatenate((X_y_c1, X_y_c2))
                np.random.shuffle(X_y_c1_c2)
                cls = clone(self.base_classifier)
                cls.fit(X_y_c1_c2[:, :-1], X_y_c1_c2[:, -1])
                self.classifiers[c1][c2] = cls

        return self

    def predict(self, X):
        check_is_fitted(self, 'n_features')
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError('Number of features is different from training phase')

        votes = []
        for c1 in self.classes:
            for c2 in self.classes:
                if c1 >= c2:
                    continue
                cls_prediction = self.classifiers[c1][c2].predict(X)
                votes.append(cls_prediction)

        votes = np.array(votes).astype(int)
        votes = votes.T
        predictions = []
        for i in range(votes.shape[0]):
            bins = np.bincount(votes[i])  # TODO count only votes with high probability
            predictions.append(np.argmax(bins))

        return np.array(predictions)