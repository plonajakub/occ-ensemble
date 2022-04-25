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
from occ_naive_bayes import OCCNaiveBayes


def search_stock_estimator(estimator, params, X, y, n_splits, scoring, results_path, ):
    gs = GridSearchCV(estimator, params, scoring=scoring,
                      cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=None),
                      n_jobs=-1, verbose=4, return_train_score=True)
    gs.fit(X, y)
    results_df = pd.DataFrame(gs.cv_results_)
    results_df.to_csv(path_or_buf=results_path, float_format='%.2f')


def get_data(n_features):
    data = pd.read_excel('../data/CTG.xls', sheet_name='Data', header=1, usecols='K:AE,AR,AT', nrows=2126)
    all_features_anova = ['DL.1', 'AC.1', 'ALTV', 'DP.1', 'Mean', 'Variance', 'Width', 'Min', 'MSTV', 'Median', 'Mode',
                          'ASTV', 'Max', 'Nmax', 'UC.1', 'MLTV', 'LB', 'Tendency', 'Nzeros', 'DS.1', 'FM.1']
    selected_features = all_features_anova[:n_features]
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

    search_stock_estimator(
        estimator=Pipeline([*transformers, ('clf', SVC())]),
        params={
            'clf__C': [0.01, 0.1, 1, 2],
            'clf__gamma': [0.0001, 0.001, 0.01, 0.1],
            # 'clf__class_weight': [None, 'balanced'],
            # 'clf__break_ties': [False, True],
        }, X=X, y=y, n_splits=cross_validation_n_splits, scoring='balanced_accuracy',
        results_path='../results/parameter_search/svc__grid_search__ba_score.csv')

    search_stock_estimator(
        estimator=Pipeline([*transformers, ('clf', OCCNaiveBayes())]),
        params={
            'clf__data_contamination': [0.1, 0.2, 0.3, 0.4, 0.5],
        }, X=X, y=y, n_splits=cross_validation_n_splits, scoring='balanced_accuracy',
        results_path='../results/parameter_search/occ_nb__grid_search__ba_score.csv')


if __name__ == '__main__':
    main()
