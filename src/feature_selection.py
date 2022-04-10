import pandas as pd
import matplotlib.pyplot as plt

from sklearn import clone
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
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
from occ_nearest_mean import OCCNearestMean
from misc import *


def main():
    data = pd.read_excel('../data/CTG.xls', sheet_name='Data', header=1, usecols='K:AE,AR,AT', nrows=2126)
    all_features_anova = ['DL.1', 'AC.1', 'ALTV', 'DP.1', 'Mean', 'Variance', 'Width', 'Min', 'MSTV', 'Median', 'Mode',
                          'ASTV', 'Max', 'Nmax', 'UC.1', 'MLTV', 'LB', 'Tendency', 'Nzeros', 'DS.1', 'FM.1']
    all_features_mi = ['Variance', 'AC.1', 'DL.1', 'MSTV', 'Width', 'Min', 'ASTV', 'Mean', 'ALTV', 'Mode', 'Median',
                       'Max', 'LB', 'MLTV', 'Nmax', 'DP.1', 'FM.1', 'UC.1', 'Nzeros', 'Tendency', 'DS.1']
    class_feature = ['CLASS']
    all_features = all_features_anova
    save_path = '../results/feature_selection/feature_selection_anova_results.csv'

    clfs = {
        'occ_max_dist': OCCEnsemble(base_classifier=svm.OneClassSVM(nu=0.015, gamma=0.2)),
        'svc': SVC(C=2, gamma=0.04, class_weight='balanced', break_ties=True),
        'occ_nearest_mean': OCCNearestMean(resolve_classifier=KNeighborsClassifier(n_neighbors=5), outlier_ratio=0.5),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'gnb': GaussianNB(),
    }

    transformers = [
        ('scaler', StandardScaler()),
        ('resampler', SMOTETomek(n_jobs=-1)),
        # ('resampler', SMOTETomek(n_jobs=-1,
        #                          sampling_strategy={1: 350, 2: 400, 3: 200, 4: 200, 5: 200,
        #                                             6: 350, 7: 350, 8: 200, 9: 200, 10: 350})),
    ]

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    results_df = pd.DataFrame()
    for n_features in range(1, len(all_features) + 1):
        print(f'Testing n_features = {n_features}...')
        data_subset = data[all_features[:n_features] + class_feature]
        data_subset_preprocessed = preprocess_data(data_subset)
        data_successfully_preprocessed = True
        # numpy_ds_bar_plot(y, 'Distribution of classes', 'Class', 'Occurrences')
        classes_count = np.bincount(data_subset_preprocessed['CLASS'])
        if np.any(classes_count[1:] < 6):  # 6 - min number of samples per class for SMOTE
            data_successfully_preprocessed = False
            X = data_subset.iloc[:, :-1].to_numpy()
            y = data_subset.iloc[:, -1].to_numpy()
        else:
            X = data_subset_preprocessed.iloc[:, :-1].to_numpy()
            y = data_subset_preprocessed.iloc[:, -1].to_numpy()
        ba_scores = {k: [] for k in clfs.keys()}
        precision_scores = {k: [] for k in clfs.keys()}
        recall_scores = {k: [] for k in clfs.keys()}
        for train_index, test_index in rskf.split(X, y):
            for clf_name, clf in clfs.items():
                clf = clone(clf)
                pipeline = Pipeline([*transformers, ('clf', clf)])
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                pipeline.fit(X_train, y_train)
                predict = pipeline.predict(X_test)
                ba_scores[clf_name].append(balanced_accuracy_score(y_test, predict))
                precision_scores[clf_name].append(precision_score(y_test, predict, average='weighted'))
                recall_scores[clf_name].append(recall_score(y_test, predict, average='weighted'))
        for clf_name in clfs.keys():
            df_item = {'n_features': [n_features],
                       'data_successfully_preprocessed': [data_successfully_preprocessed],
                       'clf': [clf_name],
                       'ba_mean': [np.mean(ba_scores[clf_name])],
                       'ba_std': [np.std(ba_scores[clf_name])],
                       'precision_mean': [np.mean(precision_scores[clf_name])],
                       'precision_std': [np.std(precision_scores[clf_name])],
                       'recall_mean': [np.mean(recall_scores[clf_name])],
                       'recall_std': [np.std(recall_scores[clf_name])]}
            results_df = pd.concat((results_df, pd.DataFrame(df_item)), axis=0, ignore_index=True)
    results_df.sort_values(by='ba_mean', inplace=True)
    results_df.to_csv(path_or_buf=save_path, float_format='%.2f')


if __name__ == '__main__':
    main()
