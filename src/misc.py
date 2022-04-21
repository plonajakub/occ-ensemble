import pandas as pd
from matplotlib.colors import Normalize
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np


def preprocess_data(df: DataFrame):
    df = df.drop_duplicates()  # regular duplicates
    df = df.drop_duplicates(subset=df.columns.values[:-1], keep=False)  # conflicting data
    # df = df.sample(frac=1).reset_index(drop=True)
    return df


def numpy_ds_bar_plot(x, title, x_label, y_label):
    x_unique, x_count = np.unique(x, return_counts=True)
    plt.bar(x_unique, height=x_count)
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
        y = clf_df_sorted['ba_mean']
        plt.xticks(np.arange(df['n_features'].min(), df['n_features'].max() + 1, 1))
        plt.plot(x, y, label=f'{clf}')
        plt.scatter(x, y)
    plt.legend()
    plt.grid()
    plt.xlabel('Liczba cech')
    plt.ylabel('Zbalansowana dokładność')
    # plt.title('Selekcja cech')
    plt.tight_layout()
    plt.show()


def plot_parameter_search_heatmap(scores_df, p1_name, p1_range, p2_name, p2_range):
    class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    scores = scores_df["mean_test_score"].to_numpy().reshape(len(p1_range), len(p2_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.hot,
        norm=MidpointNormalize(midpoint=0.65),
    )
    plt.xlabel(p2_name)
    plt.ylabel(p1_name)
    plt.colorbar()
    plt.xticks(np.arange(len(p2_range)), p2_range, rotation=45)
    plt.yticks(np.arange(len(p1_range)), p1_range)
    plt.title("Validation accuracy")
    plt.show()
    plt.close()


def main():
    svc_gs_results_df = pd.read_csv('../results/parameter_search/svc__grid_search__ba_score.csv')
    C_range = svc_gs_results_df['param_clf__C'].unique()
    C_range.sort()
    gamma_range = svc_gs_results_df['param_clf__gamma'].unique()
    gamma_range.sort()
    plot_parameter_search_heatmap(svc_gs_results_df, 'C', C_range, 'gamma', gamma_range)


if __name__ == '__main__':
    main()