import numpy as np
from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd

from occ_ensemble import OCCEnsemble


class OCCEnsembleDynamicSelection(BaseEnsemble):
    def __init__(self,
                 base_classifier=OCCEnsemble(combination='max_distance'),
                 ensemble_size=8,
                 val_size=0.25,
                 neighborhood_size=0.2,
                 max_neighborhood_size=30,
                 random_state=None, ):
        self.base_classifier = base_classifier
        self.ensemble_size = ensemble_size
        self.X_val = None
        self.y_val = None
        self.val_size = val_size
        self.neighborhood_size = neighborhood_size
        self.max_neighborhood_size = max_neighborhood_size
        self.classifiers = []
        self.random_state = random_state
        self.classes = None
        self.n_features = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        # X_train, X_val, y_train, y_val = X, None, y, None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, stratify=y)
        self.X_val = X_val
        self.y_val = y_val

        rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=self.ensemble_size // 4, random_state=self.random_state)
        for train_index, _ in rskf.split(X_train, y_train):
            ensemble_cls = clone(self.base_classifier).fit(X_train[train_index], y_train[train_index])
            self.classifiers.append(ensemble_cls)

        return self

    def predict(self, X):
        check_is_fitted(self, 'n_features')
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError('Number of features is different from training phase')

        nghd_size = np.ceil(self.X_val.shape[0] * self.neighborhood_size)
        nghd_size = np.clip([nghd_size], a_min=1, a_max=self.max_neighborhood_size)[0]
        nghd_size = int(nghd_size)

        predictions = []
        for idx in range(X.shape[0]):
            distances = euclidean_distances(self.X_val, X[np.newaxis, idx])
            distances = np.squeeze(distances)
            dist_df = pd.DataFrame(data=distances)
            dist_df.sort_values(by=0, inplace=True)
            nghd_idxs = dist_df.iloc[:nghd_size, :].index.values.astype(int)
            cls_accuracies = []
            for cls in self.classifiers:
                cls_accuracies.append(accuracy_score(self.y_val[nghd_idxs], cls.predict(self.X_val[nghd_idxs])))
            best_cls = self.classifiers[np.argmax(cls_accuracies)]
            predictions.append(best_cls.predict(X[np.newaxis, idx, :]))

        return predictions
