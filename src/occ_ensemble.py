import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.model_selection import train_test_split
from sklearn import svm


class OCCEnsemble(BaseEnsemble):
    def __init__(self, base_classifier=svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1), random_state=None,
                 combination='max_distance', predict_n_pick=3, train_split_size=0.2):
        self.base_classifier = base_classifier
        self.classifiers = []
        self.train_split_size = train_split_size
        self.combination = combination  # max_distance, weighted
        self.random_state = random_state
        self.classes = None
        self.n_features = None
        self.classifiers_effectiveness = None
        self.predict_n_pick = predict_n_pick

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        if self.combination == 'weighted':
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.train_split_size, stratify=y)
        else:
            X_train, X_val, y_train, y_val = X, None, y, None

        for c in self.classes:
            b_cls = clone(self.base_classifier)
            b_cls.fit(X_train[y_train == c, :])
            self.classifiers.append(b_cls)

        if self.combination == 'weighted':
            predictions = []
            for cls in self.classifiers:
                predictions.append(cls.predict(X_val))

            self.classifiers_effectiveness = [0 for _ in range(len(self.classifiers))]
            for cls_idx, pred in enumerate(predictions):
                for p, y_val_item in zip(pred, y_val):
                    if y_val_item == cls_idx + 1 and p == 1:
                        self.classifiers_effectiveness[cls_idx] += 1

        return self

    def predict(self, X):
        check_is_fitted(self, 'n_features')
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError('Number of features is different from training phase')

        signed_distances = []
        for cls in self.classifiers:
            signed_distances.append(cls.decision_function(X))
        signed_distances = np.array(signed_distances)
        if self.classifiers_effectiveness is not None:
            y_pred = []
            for x_idx in range(X.shape[0]):
                cls_distances = [(i, dist) for i, dist in enumerate(signed_distances[:, x_idx])]
                cls_distances.sort(key=lambda t: t[1], reverse=True)
                most_prob_classes = cls_distances[:self.predict_n_pick]
                pred_cls = max(most_prob_classes, key=lambda t: self.classifiers_effectiveness[t[0]])
                y_pred.append(self.classes[pred_cls[0]])
            return np.array(y_pred)
        else:
            predictions = np.argmax(signed_distances, axis=0)
            return self.classes[predictions]
