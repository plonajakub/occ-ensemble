import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from occ_ensemble import OCCEnsemble
from occ_ensamble2 import OCCEnsemble2
from occ_ensamble_dynamic_cls_selection import OCCEnsembleDynamicSelection
from binary_decomposition import BinaryDecompositionEnsemble
from occ_svm_max import OCCSVMMax
from occ_nearest_mean import OCCNearestMean
from occ_naive_bayes import OCCNaiveBayes
from misc import preprocess_data
from misc import resampling_multipliers as r_mltps


def main():
    results_save_dir = '../results/experiments/'

    data = pd.read_excel('../data/CTG.xls', sheet_name='Data', header=1, usecols='K:AE,AR,AT', nrows=2126)
    all_features_anova = ['DL.1', 'AC.1', 'DP.1', 'ALTV', 'Variance', 'Mean', 'Width', 'Min', 'MSTV', 'Median', 'Mode',
                          'ASTV', 'Nmax', 'Max', 'UC.1', 'MLTV', 'LB', 'Tendency', 'Nzeros', 'DS.1', 'FM.1']
    all_features_mi = ['Width', 'Variance', 'Min', 'AC.1', 'DL.1', 'MSTV', 'ASTV', 'Mean', 'Max', 'Mode', 'ALTV',
                       'Median', 'LB', 'Nmax', 'MLTV', 'DP.1', 'UC.1', 'FM.1', 'Nzeros', 'Tendency', 'DS.1']
    selected_features = all_features_mi[:17]
    sf_f2int = {f: i for i, f in enumerate(selected_features)}
    class_feature = ['CLASS']
    data = data[selected_features + class_feature]
    data = preprocess_data(data)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    # numpy_ds_bar_plot(y, 'Distribution of classes', 'Class', 'Occurrences')

    features_divisions = [
        # list(map(lambda f: sf_f2int[f],
        #          ['Variance', 'AC.1', 'DL.1', 'MSTV', 'Width', 'Min', 'Mean', 'ASTV', 'ALTV', 'Mode', 'Max', 'LB', 'Median'])),
        # list(map(lambda f: sf_f2int[f],
        #          ['DL.1', 'AC.1', 'ALTV', 'Mean', 'Variance', 'Width', 'Min', 'MSTV', 'Median', 'Mode', 'ASTV'])),

        # list(map(lambda f: sf_f2int[f],
        #          ['DL.1', 'AC.1', 'ALTV', 'DP.1', 'MSTV', 'ASTV', 'UC.1', 'MLTV', 'FM.1'])),
        # list(map(lambda f: sf_f2int[f],
        #          ['DL.1', 'AC.1', 'ALTV', 'DP.1', 'MSTV', 'ASTV', 'UC.1', 'MLTV', 'LB'])),
    ]
    clfs = {
        'occ_svm_max': OCCSVMMax(svm_nu=0.015, svm_gamma=0.2),
        'svc': SVC(C=2, gamma=0.04, class_weight='balanced', break_ties=True),
        'occ_nearest_mean': OCCNearestMean(knn_neighbors=5, data_contamination=0.1),
        'nc': NearestCentroid(),
        'occ_nb': OCCNaiveBayes(data_contamination=0),
        'gnb': GaussianNB(),
        # 'occ_max_dist': OCCEnsemble(base_classifier=svm.OneClassSVM(nu=0.015, gamma=0.2)),
        # 'binary_decomposition': BinaryDecompositionEnsemble(),
        # 'occ_ensamble_2': OCCEnsemble2(ensemble_size=5, features_divisions=features_divisions, combination_type='max'),
        # 'occ_dynamic_selection': OCCEnsembleDynamicSelection(),
        # 'occ_weighted': OCCEnsemble(combination='weighted', predict_n_pick=2, train_split_size=0.2),
        # 'occ_classifier_knn': OCCEnsemble(combination='classifier', train_split_size=0.4,
        #                                   combination_classifier=KNeighborsClassifier(n_neighbors=4)),
        # 'occ_classifier_mlp': OCCEnsemble(combination='classifier', train_split_size=0.5,
        #                                   combination_classifier=MLPClassifier(hidden_layer_sizes=(100),
        #                                                                        activation='relu', max_iter=10000)),
        # 'mlp': MLPClassifier(hidden_layer_sizes=(40), activation='logistic', max_iter=2000, learning_rate='adaptive')
    }

    n_splits = 5
    n_repeats = 2

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
    ba_scores = {k: [] for k in clfs.keys()}
    f1_scores_single = {k: [] for k in clfs.keys()}
    f1_scores_multi = {k: [] for k in clfs.keys()}
    precision_scores_single = {k: [] for k in clfs.keys()}
    precision_scores_multi = {k: [] for k in clfs.keys()}
    recall_scores_single = {k: [] for k in clfs.keys()}
    recall_scores_multi = {k: [] for k in clfs.keys()}
    confusion_matrices = {k: [] for k in clfs.keys()}

    for train_index, test_index in rskf.split(X, y):
        y_counts = np.bincount(y[train_index])
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
        for clf_name, clf in clfs.items():
            clf = clone(clf)
            pipeline = Pipeline([*transformers, ('clf', clf)])
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pipeline.fit(X_train, y_train)
            predict = pipeline.predict(X_test)
            ba_scores[clf_name].append(balanced_accuracy_score(y_test, predict))
            f1_scores_single[clf_name].append(f1_score(y_test, predict, average='macro'))
            f1_scores_multi[clf_name].append(f1_score(y_test, predict, average=None))
            precision_scores_single[clf_name].append(precision_score(y_test, predict, average='macro'))
            precision_scores_multi[clf_name].append(precision_score(y_test, predict, average=None))
            recall_scores_single[clf_name].append(recall_score(y_test, predict, average='macro'))
            recall_scores_multi[clf_name].append(recall_score(y_test, predict, average=None))
            confusion_matrices[clf_name].append(confusion_matrix(y_test, predict))

    total_folds = n_splits * n_repeats
    assert total_folds == len(ba_scores[list(clfs.keys())[0]])

    scores_to_save_db = {
        'f1': f1_scores_single,
        'balanced_accuracy': ba_scores,
        'precision': precision_scores_single,
        'recall': recall_scores_single,
    }
    results_df = pd.DataFrame()
    for fold_idx in range(total_folds):
        for score_name in scores_to_save_db.keys():
            df_item = {'score_name': [score_name], 'fold_idx': [fold_idx]}
            for clf_name in clfs.keys():
                df_item[clf_name] = [scores_to_save_db[score_name][clf_name][fold_idx]]
            results_df = pd.concat((results_df, pd.DataFrame(df_item)), axis=0, ignore_index=True)
    results_df.sort_values(by='score_name', inplace=True)
    results_df.to_csv(path_or_buf=f'{results_save_dir}/test_results_db.csv', float_format='%.4f')

    scores_to_save_simple = {
        'f1': f1_scores_single,
        'balanced_accuracy': ba_scores,
        'precision': precision_scores_single,
        'recall': recall_scores_single,
    }
    for score_name, score_values in scores_to_save_simple.items():
        results_df_simple = pd.DataFrame(score_values)
        results_df_simple.to_csv(path_or_buf=f'{results_save_dir}/test_results_simple_{score_name}.csv',
                                 float_format='%.4f')

    scores_to_save_multiclass = {
        'f1': f1_scores_multi,
        'precision': precision_scores_multi,
        'recall': recall_scores_multi,
    }
    for score_name, score_values in scores_to_save_multiclass.items():
        results_df_multiclass = pd.DataFrame()
        for cls_name, fold_class_matrix in score_values.items():
            fold_class_matrix_np = np.array(fold_class_matrix)
            for class_id in range(fold_class_matrix_np.shape[1]):
                results_df_multiclass[f'{cls_name}_{class_id + 1}'] = fold_class_matrix_np[:, class_id]
        results_df_multiclass.to_csv(path_or_buf=f'{results_save_dir}/test_results_multiclass_{score_name}.csv',
                                     float_format='%.4f')

    for clf_name in clfs.keys():
        mean_f1_score = np.mean(f1_scores_single[clf_name])
        std_f1_score = np.std(f1_scores_single[clf_name])
        print(f'{clf_name} - f1: %.3f +- %.3f' % (mean_f1_score, std_f1_score))

        mean_ba_score = np.mean(ba_scores[clf_name])
        std_ba_score = np.std(ba_scores[clf_name])
        print(f'{clf_name} - balanced accuracy: %.3f +- %.3f' % (mean_ba_score, std_ba_score))

        mean_precision_score = np.mean(precision_scores_multi[clf_name], axis=0)
        std_precision_score = np.std(precision_scores_multi[clf_name], axis=0)
        print('Average precision: %.2f +- %.2f' % (
            np.mean(precision_scores_multi[clf_name]), np.std(precision_scores_multi[clf_name])))
        print('Precision in classes: ')
        for c_idx, (m, s) in enumerate(zip(mean_precision_score, std_precision_score)):
            print(f'{c_idx + 1}: %.2f +- %.2f ' % (m, s), end='\n')
        print()

        mean_recall_score = np.mean(recall_scores_multi[clf_name], axis=0)
        std_recall_score = np.std(recall_scores_multi[clf_name], axis=0)
        print('Average recall: %.2f +- %.2f' % (
            np.mean(recall_scores_multi[clf_name]), np.std(recall_scores_multi[clf_name])))
        print('Recall in classes: ')
        for c_idx, (m, s) in enumerate(zip(mean_recall_score, std_recall_score)):
            print(f'{c_idx + 1}: %.2f +- %.2f ' % (m, s), end='\n')
        print()

        cm = confusion_matrices[clf_name][0]
        np.savetxt(f'../results/confusion_matrices/confussion_matrix__{clf_name}.csv', cm, delimiter=',', fmt='%d')
        print(f'{clf_name} - confusion matrix (first split): \n{cm}')
        print()


if __name__ == '__main__':
    main()
