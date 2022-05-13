from scipy.stats import wilcoxon, iqr
import pandas as pd
import numpy as np


def do_wilcoxon_test_simple(df_path, series_1_name, series_2_name):
    df = pd.read_csv(df_path)

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


def do_wilcoxon_test_multiclass(df_path, series_1_name, series_2_name, metric):
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

        df_row[f'{series_1_name}_mean'] = np.mean(cur_series_1)
        df_row[f'{series_1_name}_std'] = np.std(cur_series_1)

        df_row[f'{series_2_name}_mean'] = np.mean(cur_series_2)
        df_row[f'{series_2_name}_std'] = np.std(cur_series_2)

        appendable_row = {k: [v] for k, v in df_row.items()}
        results_df = pd.concat((results_df, pd.DataFrame(appendable_row)), axis=0, ignore_index=True)

    print(results_df)
    results_df.to_csv(f'../results/statistics/wilcoxon__{series_1_name}__{series_2_name}__{metric}__multiclass.csv',
                      float_format='%.3e'
                      )


def main():
    do_wilcoxon_test_simple('../results/experiments/test_results_simple_f1.csv', 'occ_svm_max', 'svc')
    do_wilcoxon_test_simple('../results/experiments/test_results_simple_f1.csv', 'occ_nearest_mean', 'nc')
    do_wilcoxon_test_simple('../results/experiments/test_results_simple_f1.csv', 'occ_nb', 'gnb')

    do_wilcoxon_test_multiclass('../results/experiments/test_results_multiclass_f1.csv', 'occ_svm_max', 'svc', 'f1')
    do_wilcoxon_test_multiclass('../results/experiments/test_results_multiclass_f1.csv', 'occ_nearest_mean', 'nc', 'f1')
    do_wilcoxon_test_multiclass('../results/experiments/test_results_multiclass_f1.csv', 'occ_nb', 'gnb', 'f1')


if __name__ == '__main__':
    main()
