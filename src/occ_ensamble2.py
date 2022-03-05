import numpy as np
from sklearn import svm
from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class OCCEnsemble2(BaseEnsemble):
    def __init__(self,
                 base_classifier=svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.01),
                 features_divisions=None,
                 ensemble_size=10,
                 val_size=0.2,
                 combination_classifier=svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.01),
                 random_state=None, ):
        self.base_classifier = base_classifier
        self.features_divisions = features_divisions
        self.val_size = val_size
        self.ensemble_size = ensemble_size
        self.classifiers = {}
        self.combination_classifier = combination_classifier
        self.combination_classifiers = {}
        self.random_state = random_state
        self.classes = None
        self.n_features = None

    def _get_class_prediction(self, c, X):
        decision_function_values = []
        cls_id = 0
        for feature_set in self.features_divisions:
            for _ in range(self.ensemble_size):
                decision_function_values.append(self.classifiers[c][cls_id].decision_function(X[:, feature_set]))
                cls_id += 1
        return np.array(decision_function_values)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        self.classifiers = {c: [] for c in self.classes}

        # X_train, X_val, y_train, y_val = X, None, y, None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, stratify=y)

        for c in self.classes:
            for feature_set in self.features_divisions:
                for _ in range(self.ensemble_size):
                    b_cls = clone(self.base_classifier)
                    X_train_class = X_train[y_train == c, :][:, feature_set]
                    sample_indexes = np.random.randint(0, X_train_class.shape[0] - 1, round(X_train_class.shape[0] / 2))
                    X_train_class_bg = X_train_class[sample_indexes]
                    b_cls.fit(X_train_class_bg)
                    self.classifiers[c].append(b_cls)

        for c in self.classes:
            distances = []
            cls_id = 0
            for feature_set in self.features_divisions:
                for _ in range(self.ensemble_size):
                    distances.append(self.classifiers[c][cls_id].decision_function(X_val[:, feature_set]))
                    cls_id += 1
            distances = np.array(distances).T
            self.combination_classifiers[c] = clone(self.combination_classifier).fit(distances, y_val)

        return self

    def predict(self, X):
        check_is_fitted(self, 'n_features')
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError('Number of features is different from training phase')

        signed_distances = []
        for c in self.classes:
            signed_distances.append(self._get_class_prediction(c, X))
        signed_distances = np.array(signed_distances)
        signed_distances = np.max(signed_distances, axis=1)
        predictions = np.argmax(signed_distances, axis=0)
        return self.classes[predictions]
