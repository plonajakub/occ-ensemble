import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

from occ_ensemble import OCCEnsemble
from occ_ensamble2 import OCCEnsemble2
from occ_ensamble_dynamic_cls_selection import OCCEnsembleDynamicSelection
from binary_decomposition import BinaryDecompositionEnsemble
from occ_nearest_mean import OCCNearestMean
from occ_naive_bayes import OCCNaiveBayes
from misc import *


def main():
    data = pd.read_excel('../data/CTG.xls', sheet_name='Data', header=1, usecols='K:AE,AR,AT', nrows=2126)
    # selected_features = ['DL.1', 'AC.1', 'ALTV', 'DP.1', 'Mean', 'MSTV', 'ASTV', 'UC.1', 'MLTV', 'LB', 'Variance',
    #                      'Mode', 'Width', 'Min', 'Max', 'Median', 'DS.1', 'FM.1']
    selected_features = ['DL.1', 'AC.1', 'ALTV', 'DP.1', 'Mean', 'MSTV', 'ASTV', 'UC.1', 'MLTV', 'LB', 'DS.1', 'FM.1']
    sf_f2int = {f: i for i, f in enumerate(selected_features)}
    class_feature = ['CLASS']
    data = data[selected_features + class_feature]
    data = preprocess_data(data)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    # numpy_ds_bar_plot(y, 'Distribution of classes', 'Class', 'Occurrences')

    features_divisions = [
        # list(map(lambda f: sf_f2int[f],
        #          ['Variance', 'AC.1', 'DL.1', 'MSTV', 'Width', 'Min', 'Mean', 'ASTV', 'ALTV', 'Mode', 'Max', 'LB', 'Median'])),
        # list(map(lambda f: sf_f2int[f],
        #          ['DL.1', 'AC.1', 'ALTV', 'Mean', 'Variance', 'Width', 'Min', 'MSTV', 'Median', 'Mode', 'ASTV'])),

        list(map(lambda f: sf_f2int[f],
                 ['DL.1', 'AC.1', 'ALTV', 'DP.1', 'MSTV', 'ASTV', 'UC.1', 'MLTV', 'FM.1'])),
        list(map(lambda f: sf_f2int[f],
                 ['DL.1', 'AC.1', 'ALTV', 'DP.1', 'MSTV', 'ASTV', 'UC.1', 'MLTV', 'LB'])),
    ]
    clfs = {
        'occ_max_dist': OCCEnsemble(base_classifier=svm.OneClassSVM(nu=0.015, gamma=0.2)),
        # 'binary_decomposition': BinaryDecompositionEnsemble(),
        # 'occ_ensamble_2': OCCEnsemble2(ensemble_size=5, features_divisions=features_divisions, combination_type='max'),
        # 'occ_dynamic_selection': OCCEnsembleDynamicSelection(),
        # 'occ_weighted': OCCEnsemble(combination='weighted', predict_n_pick=2, train_split_size=0.2),
        # 'occ_classifier_knn': OCCEnsemble(combination='classifier', train_split_size=0.4,
        #                                   combination_classifier=KNeighborsClassifier(n_neighbors=4)),
        # 'occ_classifier_mlp': OCCEnsemble(combination='classifier', train_split_size=0.5,
        #                                   combination_classifier=MLPClassifier(hidden_layer_sizes=(100),
        #                                                                        activation='relu', max_iter=10000)),
        'svc': SVC(C=2, gamma=0.04, class_weight='balanced', break_ties=True),
        'occ_nearest_mean': OCCNearestMean(resolve_classifier=KNeighborsClassifier(n_neighbors=5), outlier_ratio=0.5),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'occ_nb': OCCNaiveBayes(data_contamination=0),
        'gnb': GaussianNB(),
        # 'mlp': MLPClassifier(hidden_layer_sizes=(40), activation='logistic', max_iter=2000, learning_rate='adaptive')
    }

    transformers = [
        ('scaler', StandardScaler()),
        ('resampler', SMOTETomek(n_jobs=-1)),
        # ('resampler', SMOTETomek(n_jobs=-1,
        #                          sampling_strategy={1: 350, 2: 400, 3: 200, 4: 200, 5: 200,
        #                                             6: 350, 7: 350, 8: 200, 9: 200, 10: 350})),
    ]

    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1234)
    ba_scores = {k: [] for k in clfs.keys()}
    precision_scores = {k: [] for k in clfs.keys()}
    recall_scores = {k: [] for k in clfs.keys()}
    confusion_matrices = {k: [] for k in clfs.keys()}

    for train_index, test_index in rskf.split(X, y):
        for clf_name, clf in clfs.items():
            clf = clone(clf)
            pipeline = Pipeline([*transformers, ('clf', clf)])
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pipeline.fit(X_train, y_train)
            predict = pipeline.predict(X_test)
            ba_scores[clf_name].append(balanced_accuracy_score(y_test, predict))
            precision_scores[clf_name].append(precision_score(y_test, predict, average=None))
            recall_scores[clf_name].append(recall_score(y_test, predict, average=None))
            confusion_matrices[clf_name].append(confusion_matrix(y_test, predict))

    for clf_name in clfs.keys():
        mean_ba_score = np.mean(ba_scores[clf_name])
        std_ba_score = np.std(ba_scores[clf_name])
        print(f'{clf_name} - balanced accuracy: %.3f +- %.3f' % (mean_ba_score, std_ba_score))

        mean_precision_score = np.mean(precision_scores[clf_name], axis=0)
        std_precision_score = np.std(precision_scores[clf_name], axis=0)
        print('Average precision: %.2f +- %.2f' % (
            np.mean(precision_scores[clf_name]), np.std(precision_scores[clf_name])))
        print('Precision in classes: ')
        for c_idx, (m, s) in enumerate(zip(mean_precision_score, std_precision_score)):
            print(f'{c_idx + 1}: %.2f +- %.2f ' % (m, s), end='\n')
        print()

        mean_recall_score = np.mean(recall_scores[clf_name], axis=0)
        std_recall_score = np.std(recall_scores[clf_name], axis=0)
        print('Average recall: %.2f +- %.2f' % (
            np.mean(recall_scores[clf_name]), np.std(recall_scores[clf_name])))
        print('Recall in classes: ')
        for c_idx, (m, s) in enumerate(zip(mean_recall_score, std_recall_score)):
            print(f'{c_idx + 1}: %.2f +- %.2f ' % (m, s), end='\n')
        print()

        cm = confusion_matrices[clf_name][0]
        np.savetxt(f'../results/confussion_matrix__{clf_name}.csv', cm, delimiter=',', fmt='%d')
        print(f'{clf_name} - confusion matrix (first split): \n{cm}')
        print()


if __name__ == '__main__':
    main()
