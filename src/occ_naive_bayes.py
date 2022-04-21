import numpy as np
from scipy.stats import norm
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class OCCNaiveBayes(BaseEnsemble):
    def __init__(self,
                 data_contamination=0.1,
                 random_state=None, ):
        self.data_contamination = data_contamination
        self.random_state = random_state

        self.means = []
        self.stds = []
        self.likelihood_thresholds = []
        self.classes = None
        self.n_features = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        X_train, X_val, y_train, y_val = X, None, y, None

        for c in self.classes:
            class_objects = X[y == c]
            class_means = np.mean(class_objects, axis=0)
            self.means.append(class_means)
            class_stds = np.std(class_objects, axis=0)
            for i, class_std in enumerate(class_stds):
                if np.isclose([class_std], [0])[0]:
                    class_stds[i] += 1
            self.stds.append(class_stds)

            probas = []
            for i in range(class_objects.shape[1]):
                probas.append(norm.pdf(class_objects[:, i], loc=class_means[i], scale=class_stds[i]))
            probas = np.array(probas).T

            likelihoods = np.prod(probas, axis=1)
            sorted_lkhds = np.sort(likelihoods)
            skip_idx = np.clip(np.rint(self.data_contamination * class_objects.shape[0]), a_min=0,
                               a_max=class_objects.shape[0] - 1).astype(int)
            self.likelihood_thresholds.append(sorted_lkhds[skip_idx])

        return self

    def predict(self, X):
        check_is_fitted(self, 'n_features')
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError('Number of features is different from training phase')

        likelihoods = []
        for class_idx, _ in enumerate(self.classes):
            probas = []
            for i in range(X.shape[1]):
                probas.append(norm.pdf(X[:, i], loc=self.means[class_idx][i], scale=self.stds[class_idx][i]))
            probas = np.array(probas).T
            class_likelihoods = np.prod(probas, axis=1)
            likelihoods.append(class_likelihoods)

        likelihoods_t = np.array(likelihoods).T
        threshold_diff = likelihoods_t - np.array(self.likelihood_thresholds)[np.newaxis, :]
        predictions = np.argmax(threshold_diff, axis=1)
        return self.classes[predictions]
