import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from occ_ensemble import OCCEnsemble


def numpy_ds_bar_plot(x, title, x_label, y_label):
    x_unique, x_count = np.unique(x, return_counts=True)
    plt.bar(x_unique, height=x_count)
    plt.xticks(x_unique)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()


def main():
    data = pd.read_excel('../data/CTG.xls', sheet_name='Data', header=1, usecols='K:AE,AR,AT', nrows=2126)
    # X = data.iloc[:, :-5].to_numpy()
    # X = data[['DL.1', 'AC.1', 'ALTV', 'DP.1', 'Mean', 'MSTV', 'ASTV']].to_numpy()
    X = data[['DL.1', 'AC.1', 'ALTV', 'DP.1', 'Mean', 'MSTV', 'ASTV', 'UC.1', 'MLTV', 'LB']].to_numpy()
    y = data['CLASS'].to_numpy()

    numpy_ds_bar_plot(y, 'Distribution of classes', 'Class', 'Occurrences')

    clfs = {
        'occ_max_dist': OCCEnsemble(combination='max_distance'),
        'occ_weighted': OCCEnsemble(combination='weighted', predict_n_pick=2, train_split_size=0.2),
        'occ_classifier_knn': OCCEnsemble(combination='classifier', train_split_size=0.4,
                                          combination_classifier=KNeighborsClassifier(n_neighbors=4)),
        'occ_classifier_mlp': OCCEnsemble(combination='classifier', train_split_size=0.5,
                                          combination_classifier=MLPClassifier(hidden_layer_sizes=(100),
                                                                               activation='relu', max_iter=10000)),
        'mlp': MLPClassifier(hidden_layer_sizes=(40), activation='logistic', max_iter=2000, learning_rate='adaptive')
    }

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1234)
    scores = {k: [] for k in clfs.keys()}
    confusion_matrices = {k: [] for k in clfs.keys()}

    for train_index, test_index in rskf.split(X, y):
        for clf_name, clf in clfs.items():
            clf = clone(clf)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            predict = clf.predict(X_test)
            scores[clf_name].append(balanced_accuracy_score(y_test, predict))
            confusion_matrices[clf_name].append(confusion_matrix(y_test, predict))

    for clf_name in clfs.keys():
        mean_score = np.mean(scores[clf_name])
        std_score = np.std(scores[clf_name])
        cm = confusion_matrices[clf_name][0]
        print(f'{clf_name} - balanced accuracy: %.3f +- %.3f' % (mean_score, std_score))
        print(f'{clf_name} - confusion matrix (first split): \n{cm}')
        print()


if __name__ == '__main__':
    main()
