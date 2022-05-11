import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold

from src.misc import preprocess_data
from misc import resampling_multipliers as r_mltps
from occ_naive_bayes import OCCNaiveBayes
from occ_svm_max import OCCSVMMax
from occ_nearest_mean import OCCNearestMean


def search_stock_estimator(estimator, params, X, y, n_splits, scoring, results_path, ):
    gs = GridSearchCV(estimator, params, scoring=scoring,
                      cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=None),
                      n_jobs=-1, verbose=4, return_train_score=True)
    gs.fit(X, y)
    results_df = pd.DataFrame(gs.cv_results_)
    results_df.to_csv(path_or_buf=results_path, float_format='%.2f')


def get_data(n_features):
    data = pd.read_excel('../data/CTG.xls', sheet_name='Data', header=1, usecols='K:AE,AR,AT', nrows=2126)
    all_features_anova = ['DL.1', 'AC.1', 'DP.1', 'ALTV', 'Variance', 'Mean', 'Width', 'Min', 'MSTV', 'Median', 'Mode',
                          'ASTV', 'Nmax', 'Max', 'UC.1', 'MLTV', 'LB', 'Tendency', 'Nzeros', 'DS.1', 'FM.1']
    all_features_mi = ['Width', 'Variance', 'Min', 'AC.1', 'DL.1', 'MSTV', 'ASTV', 'Mean', 'Max', 'Mode', 'ALTV',
                       'Median', 'LB', 'Nmax', 'MLTV', 'DP.1', 'UC.1', 'FM.1', 'Nzeros', 'Tendency', 'DS.1']
    selected_features = all_features_mi[:n_features]
    class_feature = ['CLASS']
    data = data[selected_features + class_feature]
    data = preprocess_data(data)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    return X, y


def main():
    X, y = get_data(n_features=17)

    cross_validation_n_splits = 5
    training_data_ratio = (cross_validation_n_splits - 1) / cross_validation_n_splits

    y_counts = np.bincount(y).astype(float)
    y_counts *= training_data_ratio

    transformers = [
        ('scaler', StandardScaler()),
        ('undersampler',
         RandomUnderSampler(sampling_strategy={
             1: int(r_mltps[1] * y_counts[1]),
             2: int(r_mltps[2] * y_counts[2]),
             6: int(r_mltps[6] * y_counts[6]),
         }, random_state=1234)),
        ('oversampler',
         SMOTE(sampling_strategy={
             3: int(r_mltps[3] * y_counts[3]),
             4: int(r_mltps[4] * y_counts[4]),
             5: int(r_mltps[5] * y_counts[5]),
             8: int(r_mltps[8] * y_counts[8]),
             9: int(r_mltps[9] * y_counts[9]),
             10: int(r_mltps[10] * y_counts[10]),
         }, random_state=1234, n_jobs=-1))
    ]

    search_stock_estimator(
        estimator=Pipeline([*transformers, ('clf', OCCSVMMax())]),
        params={
            'clf__svm_nu': np.round(np.linspace(0.1, 1, 10), decimals=3),  # Should be in the interval (0, 1]
            'clf__svm_gamma': np.logspace(-9, 3, 13),  # > 0
        }, X=X, y=y, n_splits=cross_validation_n_splits, scoring='f1_macro',
        results_path='../results/parameter_search/occ_svm_max__grid_search__f1_score.csv')

    search_stock_estimator(
        estimator=Pipeline([*transformers, ('clf', SVC(break_ties=True))]),
        params={
            'clf__C': np.logspace(-2, 10, 13),  # > 0
            'clf__gamma': np.logspace(-9, 3, 13),
        }, X=X, y=y, n_splits=cross_validation_n_splits, scoring='f1_macro',
        results_path='../results/parameter_search/svc__grid_search__f1_score.csv')

    search_stock_estimator(
        estimator=Pipeline([*transformers, ('clf', OCCNearestMean(combination_type='knn'))]),
        params={
            'clf__knn_neighbors': np.arange(1, 11, 1, dtype=int),  # > 0
            'clf__data_contamination': np.round(np.arange(0, 0.5, 0.05, dtype=float), decimals=3),  # in [0, 1)
        }, X=X, y=y, n_splits=cross_validation_n_splits, scoring='f1_macro',
        results_path='../results/parameter_search/occ_nm_knn__grid_search__f1_score.csv')

    search_stock_estimator(
        estimator=Pipeline([*transformers, ('clf', OCCNearestMean(combination_type='max'))]),
        params={
            'clf__data_contamination': np.round(np.arange(0, 0.5, 0.01, dtype=float), decimals=3),  # in [0, 1)
        }, X=X, y=y, n_splits=cross_validation_n_splits, scoring='f1_macro',
        results_path='../results/parameter_search/occ_nm_max__grid_search__f1_score.csv')

    search_stock_estimator(
        estimator=Pipeline([*transformers, ('clf', OCCNaiveBayes())]),
        params={
            'clf__data_contamination': np.round(np.arange(0, 0.5, 0.01, dtype=float), decimals=3),  # in [0, 1)
        }, X=X, y=y, n_splits=cross_validation_n_splits, scoring='f1_macro',
        results_path='../results/parameter_search/occ_nb__grid_search__f1_score.csv')


if __name__ == '__main__':
    main()
