from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

root_path = '/ibgpu01/lmh/multi-omics/'

def read_csv(path):
    df_chunk = pd.read_csv(path, sep='\t', index_col=0, chunksize=1000)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res = pd.concat(res_chunk)
    return res

def plot_roc_multiclass(y_test, y_pred_proba, filename="", n_classes=2, var_names=['CMML', 'MDS'], colors=None, dmat=False):
    """
    Plot multi-class ROC curve for a fitted model
    Parameters
    ----------
    model: object
        Fitted model with method predict_proba
    X_test: numpy.ndarray
        Testing input matrices
    y_test: numpy.ndarray
        Test output vector
    name: String, default=""
        If a name is given, save plot with this name
    n_classes: Int
        Number of class labels
    var_names: list
        List with all variables name
    dmat: Boolean
        Wheter or not model is xgb-type
    """

    y_score = y_pred_proba

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        y_binarized = (y_test == i)
        y_scores_i = y_score[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_binarized, y_scores_i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # Plot all ROC curves
    plt.figure()

    ## plot macro-avg roc curve
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (AUC = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy',
             linestyle=':',
             linewidth=2)

    # random_color = lambda:[np.random.rand() for _ in range(3)]
    # colors = [random_color() for _ in range(n_classes)]
    # colors = ["red", "green", "blue", "magenta"]

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='{0} (AUC = {1:0.2f})'
                 ''.format(var_names[i], roc_auc[i]))


    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for multi-class model {}'.format(filename))
    plt.legend(loc="lower right")
    plt.show()

def read_feature_DI_sel(dna_methylation, gene_expression, labels, l=10, i=1, l1 = 0, l2 = 0):

    fold_samples = pd.read_csv('./data/sub_dataset/subsamples_idx_' + str(i) + 'fold.tsv', sep='\t', index_col=0)
    fold_samples = fold_samples.to_numpy().squeeze()

    dna_methylation = dna_methylation.filter(items=fold_samples, axis=1)
    gene_expression = gene_expression.filter(items=fold_samples, axis=1)
    labels = labels.filter(items=fold_samples, axis=0)

    labels = labels.to_numpy().squeeze()

    dna_methylation = dna_methylation.to_numpy()
    gene_expression = gene_expression.to_numpy()

    importance_DI = pd.read_csv('./results/importances_fold'+ str(i) + '.tsv',
                                sep='\t')

    importance_DI = importance_DI.to_numpy()

    importance_met = importance_DI[:, 0]
    importance_exp = importance_DI[:, 1]

    idx_met = np.argsort(importance_met)
    idx_exp = np.argsort(importance_exp)

    if l1 + l2 == l:
        idx_met = idx_met[-l1:]
        idx_exp = idx_exp[-l2:]
    else:
        idx_met = idx_met[-l//2:]
        idx_exp = idx_exp[-l//2:]

    dna_methylation = np.transpose(dna_methylation[idx_met, :])
    gene_expression = np.transpose(gene_expression[idx_exp, :])

    X = np.concatenate([dna_methylation, gene_expression], axis=1)
    y = labels

    return X, y

def read_feature_DI_cat(dna_methylation, gene_expression, labels, l=10,i=1):

    # 5fold cross validation
    fold_samples = pd.read_csv(root_path + 'processed/subdatasets/subsamples_idx_' + str(i) + 'fold.tsv', sep='\t', index_col=0)
    fold_samples = fold_samples.to_numpy().squeeze()

    dna_methylation = dna_methylation.filter(items=fold_samples, axis=1)
    gene_expression = gene_expression.filter(items=fold_samples, axis=1)


    labels = labels.filter(items=fold_samples, axis=0)
    labels = labels.to_numpy().squeeze()

    ## MODFR
    importances = pd.read_csv('/ibgpu01/lmh/multi-omics/database/3.28/MODFR_3_dna_300_fold' + str(i) + '.tsv', sep='\t')
    importances = importances.to_numpy().squeeze()

    dna_methylation = dna_methylation.to_numpy()
    idx = np.argsort(importances)
    dna_methylation = dna_methylation[idx[-l//2:], ]

    importances = pd.read_csv('/ibgpu01/lmh/multi-omics/database/3.28/MODFR_3_rna_300_fold' + str(i) + '.tsv', sep='\t')
    importances = importances.to_numpy().squeeze()

    gene_expression = gene_expression.to_numpy()
    idx = np.argsort(importances)
    gene_expression = gene_expression[idx[-l//2:], ]

    dna_methylation = dna_methylation.transpose()
    gene_expression = gene_expression.transpose()

    X = np.concatenate([dna_methylation, gene_expression], axis=1)
    y = labels
    return X, y

def train_classifier(X, y, classifier='svm'):
    print(X.shape)
    print(y.shape)

    ## 5fold_cross_validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(probability=True,
                  gamma='scale',
                  C=10,
                  kernel='rbf',
                  tol=1e-3,
                  )

        clf.fit(X_train, y_train)
        one_hot = OneHotEncoder(sparse=False).fit(y_train.reshape(-1, 1))
        auc = roc_auc_score(one_hot.transform(y_test.reshape(-1, 1)), clf.predict_proba(X_test), multi_class='ovo', average='macro')

        # plot_roc_multiclass(y_test, clf.predict_proba(X_test), n_classes=4, var_names=['LumA', 'LumB', 'Basal', 'Her2'])

        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')

        # print('acc:{:.02f},precision:{:.03f},recall:{:.3f},f1_score:{:.3f}'.format(accuracy,
        #                                                                            precision,
        #                                                                            recall,
        #                                                                            f1))

        acc_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)

    acc_ave = np.mean(acc_list)
    precision_ave = np.mean(precision_list)
    recall_ave = np.mean(recall_list)
    f1_ave = np.mean(f1_list)
    auc_ave = np.mean(auc_list)


    acc_std = np.std(acc_list)
    precision_std = np.std(precision_list)
    recall_std = np.std(recall_list)
    f1_std = np.std(f1_list)
    auc_std = np.std(auc_list)

    print('acc{:.3f}±{:.3f}%'.format(acc_ave, acc_std))
    print('pre{:.3f}±{:.3f}'.format(precision_ave, precision_std))
    print('recall{:.3f}±{:.3f}'.format(recall_ave, recall_std))
    print('f1{:.3f}±{:.3f}'.format(f1_ave, f1_std))
    print('auc{:.3f}±{:.3f}'.format(auc_ave, auc_std))

    return acc_ave, acc_std


if __name__ == '__main__':
    '''
        Di omics
    '''


    labels = pd.read_csv(root_path + 'processed/subdatasets/subsamples_labels_nonormal.tsv', sep='\t', index_col=0)
    filter_samples = pd.read_csv(root_path + 'processed/subdatasets/subsamples_idx_nonormal.tsv', sep='\t', index_col=0)
    filter_list = filter_samples.to_numpy().squeeze()
    dna_methylation = pd.read_csv(root_path + 'database/differential cpgs/dna_methylation_co10.tsv', sep='\t',
                                  index_col=0)
    dna_methylation = dna_methylation.filter(items=filter_list, axis=1)
    print(dna_methylation.shape)
    gene_expression = pd.read_csv(root_path + 'database/differential genes/gene_expression_co10_5127.tsv', sep='\t',
                                  index_col=0)
    print(gene_expression.shape)

    ls = np.arange(10, 102, 30)
    acc_global = [[] for _ in range(5)]

    for i in range(1, 5+1, 1):
        print('fold:', i)
        acc = []
        s = 'sel'
        # s = 'cat'
        for l in ls:
            if s == 'sel':
                X, y = read_feature_DI_sel(dna_methylation, gene_expression, labels, l=l, i=i)
            elif s == 'cat':
                X, y = read_feature_DI_cat(dna_methylation, gene_expression, labels, l=l, i=i)

            acc_cur, acc_std = train_classifier(X, y, classifier='svm')
            acc.append((acc_cur, acc_std))
            acc_global[i-1].append(acc_cur)
        acc = pd.DataFrame(data=np.array(acc), index=ls, columns=['mean', 'std'])
        print(acc)
    results = [(np.mean(c), np.std(c)) for c in zip(*acc_global)]
    results = pd.DataFrame(data=np.array(results), index=ls, columns=['acc-mean', 'acc-std'])
    print(results)




