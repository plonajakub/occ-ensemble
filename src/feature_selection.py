from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn import clone
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from misc import *
from occ_naive_bayes import OCCNaiveBayes
from occ_nearest_mean import OCCNearestMean
from occ_svm_max import OCCSVMMax


def main():
    data = pd.read_excel('../data/CTG.xls', sheet_name='Data', header=1, usecols='K:AE,AR,AT', nrows=2126)
    all_features_anova = ['DL.1', 'AC.1', 'ALTV', 'DP.1', 'Mean', 'Variance', 'Width', 'Min', 'MSTV', 'Median', 'Mode',
                          'ASTV', 'Max', 'Nmax', 'UC.1', 'MLTV', 'LB', 'Tendency', 'Nzeros', 'DS.1', 'FM.1']
    all_features_mi = ['Variance', 'AC.1', 'DL.1', 'MSTV', 'Width', 'Min', 'ASTV', 'Mean', 'ALTV', 'Mode', 'Median',
                       'Max', 'LB', 'MLTV', 'Nmax', 'DP.1', 'FM.1', 'UC.1', 'Nzeros', 'Tendency', 'DS.1']
    class_feature = ['CLASS']
    all_features = all_features_mi
    save_path = '../results/feature_selection/feature_selection_mi_results.csv'

    clfs = {
        'occ_svm_max': OCCSVMMax(svm_nu=0.015, svm_gamma=0.2),
        'svc': SVC(C=2, gamma=0.04, class_weight='balanced', break_ties=True),
        'occ_nearest_mean': OCCNearestMean(knn_neighbors=5, data_contamination=0.1),
        'nc': NearestCentroid(),
        'occ_nb': OCCNaiveBayes(data_contamination=0),
        'gnb': GaussianNB(),
    }

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
            y_counts = np.bincount(y[train_index])
            transformers = [
                ('scaler', StandardScaler()),
                ('undersampler',
                 RandomUnderSampler(sampling_strategy={
                     1: int(0.75 * y_counts[1]),
                     2: int(0.5 * y_counts[2]),
                     6: int(0.75 * y_counts[6]),
                 }, random_state=1234)),
                ('oversampler',
                 SMOTE(sampling_strategy={
                     3: int(5 * y_counts[3]),
                     4: int(3 * y_counts[4]),
                     5: int(4 * y_counts[5]),
                     8: int(2.5 * y_counts[8]),
                     9: int(4 * y_counts[9]),
                     10: int(1.3 * y_counts[10]),
                 }, random_state=1234, n_jobs=-1))
            ]
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
