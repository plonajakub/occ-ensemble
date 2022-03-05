from functools import partial

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def rate_features_anova(df):
    df_values = df.values
    features = df_values[:, 0:-1]
    labels = df_values[:, -1]
    select_k_best = SelectKBest(score_func=f_classif, k='all')
    result = select_k_best.fit(features, labels)
    return result.scores_


def rate_features_mutual_info(df, discrete_features_indexes, random_state=None):
    df_values = df.values
    features = df_values[:, 0:-1]
    labels = df_values[:, -1]
    score_func = partial(mutual_info_classif,
                         discrete_features=discrete_features_indexes, random_state=random_state)
    select_k_best = SelectKBest(score_func=score_func, k='all')
    result = select_k_best.fit(features, labels)
    return result.scores_


def print_feature_scores(features, scores, title, xlabel):
    df = pd.DataFrame({'features': features, 'scores': scores})

    sorted_df = df.sort_values(by='scores')
    y_range = range(1, len(df.index) + 1)

    plt.figure(figsize=(8, 6))
    plt.hlines(y=y_range, xmin=0, xmax=sorted_df['scores'], color='skyblue')
    plt.plot(sorted_df['scores'], y_range, "o")
    plt.grid(True)
    for (_, row), y in zip(sorted_df.iterrows(), y_range):
        plt.annotate('%.2g' % row['scores'],
                     (row['scores'] + max(scores) / 50, y - 0.15))

    plt.yticks(y_range, sorted_df['features'])
    plt.title(title, loc='left')
    plt.xlabel(xlabel)
    plt.ylabel('Feature')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = pd.read_excel('../data/CTG.xls', sheet_name='Data', header=1, usecols='K:AE,AR,AT', nrows=2126)
    all_df = data.iloc[:, :-1]

    scores_all = rate_features_anova(all_df)
    print_feature_scores(list(all_df.columns.values[:-1]), scores_all,
                         title="ANOVA", xlabel='F')

    scores_mi_all = rate_features_mutual_info(
        all_df, discrete_features_indexes=[20])  # also discrete features should be added here (not only categorical)
    print_feature_scores(list(all_df.columns.values[:-1]), scores_mi_all,
                         title="Mutual information", xlabel='mi')
