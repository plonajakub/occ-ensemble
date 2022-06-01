from scipy.stats import wilcoxon, iqr
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


def do_wilcoxon_test_simple(df_path, series_1_name, series_2_name, metric, float_length=3):
    df = pd.read_csv(df_path)
    df_dscr = df.describe()

    series_1 = df[series_1_name]
    print('###############################')
    print(series_1_name, 'statistics')
    print(series_1.describe())
    print(f'Quartile deviation: {iqr(series_1.tolist()) / 2:.3f}')
    print('###############################')

    series_2 = df[series_2_name]
    print('###############################')
    print(series_2_name, 'statistics')
    print(series_2.describe())
    print(f'Quartile deviation: {iqr(series_2.tolist()) / 2:.3f}')
    print('###############################')

    stat, p_val = wilcoxon(series_1.tolist(), series_2.tolist())
    print(f'{series_1_name} vs {series_2_name} (wilcoxon): \nstat={stat:.3e} \npval={p_val:.3e}')
    s1_median = df_dscr.loc['50%', series_1_name]
    s2_median = df_dscr.loc['50%', series_2_name]
    series_with_stat_adv = '-'
    if p_val < 0.05:
        if s1_median > s2_median:
            series_with_stat_adv = series_1_name
        else:
            series_with_stat_adv = series_2_name
    p_val_presentation = '< 0.001' if p_val < 0.001 else round(p_val, 3)
    results_dict = {'Statystyka': [stat], 'P-wartość': [p_val_presentation], 'Przewaga': [series_with_stat_adv]}
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'../results/statistics/wilcoxon_{series_1_name}_{series_2_name}_{metric}_average.csv',
                      float_format=f'%.{float_length}f', index=False)


def do_wilcoxon_test_multiclass(df_path, series_1_name, series_2_name, metric, float_length=3):
    df = pd.read_csv(df_path)
    results_df = pd.DataFrame()
    for class_name in range(1, 11):
        cur_series_1 = df[f'{series_1_name}_{class_name}']
        cur_series_2 = df[f'{series_2_name}_{class_name}']

        df_row = {'class': class_name}

        try:
            stat, p_val = wilcoxon(cur_series_1.tolist(), cur_series_2.tolist())
            test_success = True
        except ValueError:
            stat, p_val = -1, -1
            test_success = False
        df_row['stat'] = stat
        df_row['p_val'] = p_val

        df_row[f'{series_1_name}_stat_adv'] = 0
        df_row[f'{series_2_name}_stat_adv'] = 0
        alpha = 0.05
        if test_success and p_val < alpha:
            if np.median(cur_series_1) > np.median(cur_series_2):
                df_row[f'{series_1_name}_stat_adv'] = 1
            else:
                df_row[f'{series_2_name}_stat_adv'] = 1

        df_row[f'{series_1_name}_median'] = np.median(cur_series_1)
        df_row[f'{series_1_name}_qd'] = iqr(cur_series_1.tolist()) / 2

        df_row[f'{series_2_name}_median'] = np.median(cur_series_2)
        df_row[f'{series_2_name}_qd'] = iqr(cur_series_2.tolist()) / 2

        # df_row[f'{series_1_name}_mean'] = np.mean(cur_series_1)
        # df_row[f'{series_1_name}_std'] = np.std(cur_series_1)

        # df_row[f'{series_2_name}_mean'] = np.mean(cur_series_2)
        # df_row[f'{series_2_name}_std'] = np.std(cur_series_2)

        appendable_row = {k: [v] for k, v in df_row.items()}
        results_df = pd.concat((results_df, pd.DataFrame(appendable_row)), axis=0, ignore_index=True)

    p_val_new = results_df['p_val'].map(lambda x: '< 0.001' if x < 0.001 else round(x, 3))
    results_df['p_val_new'] = p_val_new

    final_column_names = {'class': 'Klasa',
                          'stat': 'Statystyka',
                          'p_val_new': 'P-wartość',
                          f'{series_1_name}_stat_adv': f'{series_1_name}Przewaga',
                          f'{series_2_name}_stat_adv': f'{series_2_name}Przewaga',
                          f'{series_1_name}_median': f'{series_1_name}Mediana',
                          f'{series_2_name}_median': f'{series_2_name}Mediana',
                          f'{series_1_name}_qd': f'{series_1_name}Odch. ćw.',
                          f'{series_2_name}_qd': f'{series_2_name}Odch. ćw.'}
    results_df = results_df[list(final_column_names.keys())]
    results_df = results_df.rename(columns=final_column_names)

    print(results_df)
    results_df.to_csv(f'../results/statistics/wilcoxon__{series_1_name}__{series_2_name}__{metric}__multiclass.csv',
                      float_format=f'%.{float_length}f', index=False)


def print_save_statistics(path, columns):
    df = pd.read_csv(path)
    df = df[columns]

    df_dsc = df.describe()
    df_dsc = df_dsc.drop('count')
    df_dsc = df_dsc.rename(
        {'mean': 'Średnia', 'std': 'Odch. std.', 'min': 'Minimum', '25%': 'Q1', '50%': 'Mediana', '75%': 'Q3',
         'max': 'Maksimum'})
    quartile_deviation = iqr(df.to_numpy(), axis=0) / 2
    df_dsc = pd.concat([df_dsc,
                        pd.DataFrame(data=quartile_deviation[np.newaxis, :], index=['Odch. ćw.'],
                                     columns=df_dsc.columns.values)])

    dirname = '../results/statistics'
    filename = os.path.basename(path).split('.')[0] + f'_stat_{"_".join(columns)}.csv'
    df_dsc.to_csv(f'{dirname}/{filename}', float_format='%.3f')


def print_boxplots(path, columns):
    df = pd.read_csv(path)
    df = df[columns]

    sns.set_theme()
    sns.boxplot(data=df, showmeans=True)

    plt.tight_layout()
    plt.savefig(f'../results/img/boxplot_{"_".join(columns)}')
    plt.close()


def print_multiclass_boxplots(path, classifiers, ylabel, classes=None):
    majority_classes = [1, 2, 6, 7]
    minority_classes = [3, 4, 5, 8, 9, 10]

    df = pd.read_csv(path)
    df = df[df['Klasyfikator'].isin(classifiers)]
    if classes is not None:
        if classes == 'minority':
            df = df[df['Klasa'].isin(minority_classes)]
        else:
            df = df[df['Klasa'].isin(majority_classes)]

    sns.set_theme()
    sns.boxplot(x='Klasa', y='Metryka', hue='Klasyfikator', data=df, showmeans=True)

    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'../results/img/boxplot_multi_{classes}_{"_".join(classifiers)}')
    plt.close()


def convert_multiclass_format(path):
    df = pd.read_csv(path)
    df_n_rows = df.shape[0]
    columns = df.columns.values[1:]
    cls_class_columns = [('_'.join(column.split('_')[:-1]), column.split('_')[-1]) for column in columns]
    converted_df = pd.DataFrame()
    for col, (classifier, obj_class) in zip(columns, cls_class_columns):
        appendable_dict = {'Klasa': [obj_class for _ in range(df_n_rows)],
                           'Klasyfikator': [classifier for _ in range(df_n_rows)],
                           'Metryka': df[col].tolist()
                           }
        converted_df = pd.concat([converted_df, pd.DataFrame(appendable_dict)], ignore_index=True)
    dirname = '../results/statistics/converted_multiclass'
    filename = os.path.basename(path).split('.')[0] + '_db.csv'
    converted_df.to_csv(f'{dirname}/{filename}', index=False)
    pass


def main():
    do_wilcoxon_test_simple('../results/experiments/test_results_simple_f1.csv', 'occ_svm_max', 'svc', 'f1')
    do_wilcoxon_test_simple('../results/experiments/test_results_simple_f1.csv', 'occ_nearest_mean', 'nc', 'f1')
    do_wilcoxon_test_simple('../results/experiments/test_results_simple_f1.csv', 'occ_nb', 'gnb', 'f1')

    do_wilcoxon_test_multiclass('../results/experiments/test_results_multiclass_f1.csv', 'occ_svm_max', 'svc', 'f1')
    do_wilcoxon_test_multiclass('../results/experiments/test_results_multiclass_f1.csv', 'occ_nearest_mean', 'nc', 'f1')
    do_wilcoxon_test_multiclass('../results/experiments/test_results_multiclass_f1.csv', 'occ_nb', 'gnb', 'f1')

    print_save_statistics('../results/experiments/test_results_simple_f1.csv', columns=['occ_svm_max', 'svc'])
    print_save_statistics('../results/experiments/test_results_simple_f1.csv', columns=['occ_nearest_mean', 'nc'])
    print_save_statistics('../results/experiments/test_results_simple_f1.csv', columns=['occ_nb', 'gnb'])

    print_boxplots('../results/experiments/test_results_simple_f1.csv', columns=['occ_svm_max', 'svc'])
    print_boxplots('../results/experiments/test_results_simple_f1.csv', columns=['occ_nearest_mean', 'nc'])
    print_boxplots('../results/experiments/test_results_simple_f1.csv', columns=['occ_nb', 'gnb'])

    print_boxplots('../results/experiments/test_results_simple_precision.csv', columns=['occ_svm_max', 'svc'])
    print_boxplots('../results/experiments/test_results_simple_precision.csv', columns=['occ_nearest_mean', 'nc'])
    print_boxplots('../results/experiments/test_results_simple_precision.csv', columns=['occ_nb', 'gnb'])

    print_boxplots('../results/experiments/test_results_simple_recall.csv', columns=['occ_svm_max', 'svc'])
    print_boxplots('../results/experiments/test_results_simple_recall.csv', columns=['occ_nearest_mean', 'nc'])
    print_boxplots('../results/experiments/test_results_simple_recall.csv', columns=['occ_nb', 'gnb'])

    convert_multiclass_format('../results/experiments/test_results_multiclass_f1.csv')
    print_multiclass_boxplots('../results/statistics/converted_multiclass/test_results_multiclass_f1_db.csv',
                              classifiers=['occ_svm_max', 'svc'], classes=None, ylabel='F1')
    print_multiclass_boxplots('../results/statistics/converted_multiclass/test_results_multiclass_f1_db.csv',
                              classifiers=['occ_nearest_mean', 'nc'], classes=None, ylabel='F1')
    print_multiclass_boxplots('../results/statistics/converted_multiclass/test_results_multiclass_f1_db.csv',
                              classifiers=['occ_nb', 'gnb'], classes=None, ylabel='F1')

    convert_multiclass_format('../results/experiments/test_results_multiclass_precision.csv')
    print_multiclass_boxplots('../results/statistics/converted_multiclass/test_results_multiclass_precision_db.csv',
                              classifiers=['occ_svm_max', 'svc'], classes=None, ylabel='Precyzja')
    print_multiclass_boxplots('../results/statistics/converted_multiclass/test_results_multiclass_precision_db.csv',
                              classifiers=['occ_nearest_mean', 'nc'], classes=None, ylabel='Precyzja')
    print_multiclass_boxplots('../results/statistics/converted_multiclass/test_results_multiclass_precision_db.csv',
                              classifiers=['occ_nb', 'gnb'], classes=None, ylabel='Precyzja')

    convert_multiclass_format('../results/experiments/test_results_multiclass_recall.csv')
    print_multiclass_boxplots('../results/statistics/converted_multiclass/test_results_multiclass_recall_db.csv',
                              classifiers=['occ_svm_max', 'svc'], classes=None, ylabel='Czułość')
    print_multiclass_boxplots('../results/statistics/converted_multiclass/test_results_multiclass_recall_db.csv',
                              classifiers=['occ_nearest_mean', 'nc'], classes=None, ylabel='Czułość')
    print_multiclass_boxplots('../results/statistics/converted_multiclass/test_results_multiclass_recall_db.csv',
                              classifiers=['occ_nb', 'gnb'], classes=None, ylabel='Czułość')


if __name__ == '__main__':
    main()
