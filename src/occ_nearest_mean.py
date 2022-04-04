import numpy as np
from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics.pairwise import euclidean_distances


class OCCNearestMean(BaseEnsemble):
    def __init__(self,
                 resolve_classifier=KNeighborsClassifier(n_neighbors=5),
                 # decision_boundary_coef=3,
                 outlier_ratio=0.00,
                 random_state=None, ):
        self.resolve_classifier = resolve_classifier
        # self.decision_boundary_coef = decision_boundary_coef
        self.outlier_ratio = outlier_ratio
        self.random_state = random_state

        self.means = []
        self.max_distances = []  # max distances from means
        self.classes = None
        self.n_features = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.train_split_size, stratify=y)
        X_train, X_val, y_train, y_val = X, None, y, None
        self.X_train = X_train
        self.y_train = y_train

        for c in self.classes:
            class_objects = X[y == c]
            class_mean = np.mean(class_objects, axis=0)
            self.means.append(class_mean)
            # class_std = np.std(class_objects, axis=0)
            # max_distance_from_mean = self.decision_boundary_coef * class_std
            # self.max_distances.append(max_distance_from_mean)
            distances = euclidean_distances(self.X_train, class_mean[np.newaxis, :])
            distances = np.squeeze(distances)
            sorted_distances = np.sort(distances)[::-1]
            skip_idx = np.clip(np.rint(self.outlier_ratio * X_train.shape[0]), a_min=0,
                               a_max=X_train.shape[0] - 1).astype(int)
            self.max_distances.append(sorted_distances[skip_idx])

        return self

    def predict(self, X):
        check_is_fitted(self, 'n_features')
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError('Number of features is different from training phase')

        distances = euclidean_distances(X, self.means)
        boundary_offset = np.array(self.max_distances)[np.newaxis, :] - distances
        predictions = np.argmax(boundary_offset, axis=1)
        return self.classes[predictions]