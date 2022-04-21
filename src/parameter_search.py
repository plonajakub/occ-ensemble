import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.misc import preprocess_data


def search_stock_estimator(estimator, params, X, y, scoring, results_path, ):
    gs = GridSearchCV(estimator, params, scoring=scoring, n_jobs=-1, verbose=4, return_train_score=True)
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

    transformers = [
        ('scaler', StandardScaler()),
        ('resampler', SMOTETomek(n_jobs=-1)),
        # ('resampler', SMOTETomek(n_jobs=-1,
        #                          sampling_strategy={1: 350, 2: 400, 3: 200, 4: 200, 5: 200,
        #                                             6: 350, 7: 350, 8: 200, 9: 200, 10: 350})),
    ]

    search_stock_estimator(
        estimator=Pipeline([*transformers, ('clf', SVC())]),
        params={
            'clf__C': [0.01, 0.1, 1, 2],
            'clf__gamma': [0.0001, 0.001, 0.01, 0.1],
            # 'clf__class_weight': [None, 'balanced'],
            # 'clf__break_ties': [False, True],
        }, X=X, y=y, scoring='balanced_accuracy',
        results_path='../results/parameter_search/svc__grid_search__ba_score.csv')


if __name__ == '__main__':
    main()
