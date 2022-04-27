import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class OCCNearestMean(BaseEnsemble):
    def __init__(self,
                 knn_neighbors=5,
                 data_contamination=0.00,
                 combination_type='knn',
                 random_state=None, ):
        self.knn_neighbors = knn_neighbors
        self.data_contamination = data_contamination
        self.combination_type = combination_type
        self.random_state = random_state

        self.resolve_classifier = None
        self.means = []
        self.max_distances = []  # max distances from means
        self.classes = None
        self.n_features = None
        self.train_df = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        if self.combination_type not in ['max', 'knn']:
            raise ValueError(f'Possible combination types: max, knn; got {self.combination_type}')

        self.resolve_classifier = KNeighborsClassifier(n_neighbors=self.knn_neighbors, n_jobs=-1)

        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.train_split_size, stratify=y)
        X_train, X_val, y_train, y_val = X, None, y, None
        self.train_df = pd.DataFrame(data=X_train, copy=True)
        self.train_df['y'] = y_train

        for c in self.classes:
            class_objects = X_train[y_train == c]
            class_mean = np.mean(class_objects, axis=0)
            self.means.append(class_mean)
            distances = euclidean_distances(class_objects, class_mean[np.newaxis, :])
            distances = np.squeeze(distances)
            sorted_distances = np.sort(distances)[::-1]
            skip_idx = np.clip(np.rint(self.data_contamination * class_objects.shape[0]), a_min=0,
                               a_max=class_objects.shape[0] - 1).astype(int)
            self.max_distances.append(sorted_distances[skip_idx])

        return self

    def predict(self, X):
        check_is_fitted(self, 'n_features')
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError('Number of features is different from training phase')

        distances = euclidean_distances(X, self.means)
        boundary_offset = np.array(self.max_distances)[np.newaxis, :] - distances

        assert self.combination_type in ['knn', 'max']
        if self.combination_type == 'knn':
            predictions = []
            for dfv, prd_obj in zip(boundary_offset, X):  # dfv - decision function vector
                pred = np.argmax(dfv)

                if dfv[pred] < 0:
                    predictions.append(self.classes[pred])
                    continue

                inliers = []
                for idx, df in enumerate(dfv):
                    if df >= 0:
                        inliers.append(self.classes[idx])
                if len(inliers) == 1:
                    predictions.append(self.classes[pred])
                    continue

                tmp_resolve_cls = clone(self.resolve_classifier)
                selected_train_data = self.train_df.loc[self.train_df['y'].isin(inliers)]
                tmp_resolve_cls.fit(selected_train_data.iloc[:, :-1], selected_train_data['y'])
                resolved_pred = tmp_resolve_cls.predict(prd_obj[np.newaxis, :])
                predictions.append(resolved_pred[0])

            return predictions
        else:  # self.combination_type == 'max'
            predictions = np.argmax(boundary_offset, axis=1)
            return self.classes[predictions]
