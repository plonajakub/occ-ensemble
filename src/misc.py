import pandas as pd
from matplotlib.colors import Normalize
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

resampling_multipliers = {
    1: 0.75,
    2: 0.5,
    3: 5,
    4: 3.5,
    5: 3.7,
    6: 0.85,
    7: 1,
    8: 2.5,
    9: 3.9,
    10: 1.35
}


def preprocess_data(df: DataFrame):
    print('### Data Preprocessing ###')
    print(f'Input shape: {df.shape}')
    numpy_ds_bar_plot(df.iloc[:, -1].to_numpy(), '', 'Klasa', 'Wystąpienia')

    n_obj_before_duplicates = df.shape[0]
    df = df.drop_duplicates()  # regular duplicates
    n_obj_after_duplicates = df.shape[0]

    n_obj_before_conflicts = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.values[:-1], keep=False)  # conflicting data
    n_obj_after_conflicts = df.shape[0]

    n_obj_before_outliers = df.shape[0]
    lof = LocalOutlierFactor()
    is_inlier = lof.fit_predict(df.iloc[:, :-1])
    is_inlier[is_inlier == -1] = 0
    is_inlier = is_inlier.astype(bool)
    df = df.loc[is_inlier]
    n_obj_after_outliers = df.shape[0]

    # df = df.sample(frac=1).reset_index(drop=True)

    print(f'Removed duplicates: {n_obj_before_duplicates - n_obj_after_duplicates}')
    print(f'Removed conflicts: {n_obj_before_conflicts - n_obj_after_conflicts}')
    print(f'Removed outliers: {n_obj_before_outliers - n_obj_after_outliers}')
    print(f'Output shape: {df.shape}')
    print('##########################')
    numpy_ds_bar_plot(df.iloc[:, -1].to_numpy(), '', 'Klasa', 'Wystąpienia')
    return df


def numpy_ds_bar_plot(x, title, x_label, y_label):
    x_unique, x_count = np.unique(x, return_counts=True)
    bars = plt.bar(x_unique, height=x_count)
    counts = np.bincount(x)[1:]
    for c, rect in zip(counts, bars):
        plt.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height(), f'{rect.get_height():.0f}',
                 ha='center', va='bottom')
    plt.xticks(x_unique)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()


def plot_feature_selection_results(path):
    df = pd.read_csv(path)
    for clf in df['clf'].unique():
        clf_df = df[df['clf'] == clf]
        clf_df_sorted = clf_df.sort_values(by='n_features')
        x = clf_df_sorted['n_features']
        y = clf_df_sorted['f1_mean']
        plt.xticks(np.arange(df['n_features'].min(), df['n_features'].max() + 1, 1))
        plt.plot(x, y, label=f'{clf}')
        plt.scatter(x, y)
    plt.legend()
    plt.grid()
    plt.xlabel('Liczba cech')
    plt.ylabel('F1')
    # plt.title('Selekcja cech')
    plt.tight_layout()
    plt.show()


def plot_parameter_search_plot(scores_df_path, param_df_key, param_name, y_label):
    scores_df = pd.read_csv(scores_df_path)
    scores_df_sorted = scores_df.sort_values(by=param_df_key)
    plt.plot(scores_df_sorted[param_df_key], scores_df_sorted['mean_test_score'])
    plt.scatter(scores_df_sorted[param_df_key], scores_df_sorted['mean_test_score'])
    plt.xlabel(param_name)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()
    plt.close()


def plot_parameter_search_heatmap(scores_df_path, p1_name, p1_df_key, p2_name, p2_df_key, midpoint=0.65, scientific=False):
    """p1 changes first in data"""
    scores_df = pd.read_csv(scores_df_path)
    p1_range = scores_df[p1_df_key].unique()
    p1_range.sort()
    p2_range = scores_df[p2_df_key].unique()
    p2_range.sort()

    class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    scores = scores_df["mean_test_score"].to_numpy().reshape(len(p1_range), len(p2_range), order='F')

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.hot,
        norm=MidpointNormalize(midpoint=midpoint),
    )
    plt.xlabel(p2_name)
    plt.ylabel(p1_name)
    plt.colorbar()
    if scientific:
        p2_range_sct = [np.format_float_scientific(el, precision=1) for el in p2_range]
        p1_range_sct = [np.format_float_scientific(el, precision=1) for el in p1_range]
        plt.xticks(np.arange(len(p2_range)), p2_range_sct, rotation=45)
        plt.yticks(np.arange(len(p1_range)), p1_range_sct)
    else:
        plt.xticks(np.arange(len(p2_range)), p2_range, rotation=45)
        plt.yticks(np.arange(len(p1_range)), p1_range)
    # plt.title("Validation accuracy")
    plt.show()
    plt.close()


def main():
    # plot_feature_selection_results('../results/feature_selection/feature_selection_mi_results.csv')
    # plot_feature_selection_results('../results/feature_selection/feature_selection_anova_results.csv')

    plot_parameter_search_heatmap('../results/parameter_search/occ_svm_max__grid_search__f1_score.csv',
                                  'nu', 'param_clf__svm_nu', 'gamma', 'param_clf__svm_gamma', midpoint=0.5, scientific=True)

    plot_parameter_search_heatmap('../results/parameter_search/svc__grid_search__f1_score.csv',
                                  'gamma', 'param_clf__gamma', 'C', 'param_clf__C', midpoint=0.73, scientific=True)

    plot_parameter_search_heatmap('../results/parameter_search/occ_nm_knn__grid_search__f1_score.csv',
                                  'knn_neighbors', 'param_clf__knn_neighbors', 'data_contamination', 'param_clf__data_contamination',
                                  midpoint=0.70, scientific=False)

    plot_parameter_search_plot('../results/parameter_search/occ_nb__grid_search__f1_score.csv',
                               'param_clf__data_contamination', 'Data contamination - nb', 'F1')

    plot_parameter_search_plot('../results/parameter_search/occ_nm_max__grid_search__f1_score.csv',
                               'param_clf__data_contamination', 'Data contamination - nm_max', 'F1')


if __name__ == '__main__':
    main()
