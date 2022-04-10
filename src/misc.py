import pandas as pd
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