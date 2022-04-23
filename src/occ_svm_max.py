import numpy as np
from sklearn import svm
from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class OCCSVMMax(BaseEnsemble):
    def __init__(self,
                 svm_nu=0.5,
                 svm_gamma='scale',
                 random_state=None, ):
        self.svm_nu = svm_nu
        self.svm_gamma = svm_gamma
        self.base_classifier = svm.OneClassSVM(nu=svm_nu, gamma=svm_gamma)
        self.random_state = random_state

        self.classifiers = []
        self.classes = None
        self.n_features = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        X_train, X_val, y_train, y_val = X, None, y, None

        for c in self.classes:
            b_cls = clone(self.base_classifier)
            b_cls.fit(X_train[y_train == c])
            self.classifiers.append(b_cls)

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

        predictions = np.argmax(signed_distances, axis=0)
        return self.classes[predictions]
