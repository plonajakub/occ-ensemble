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
