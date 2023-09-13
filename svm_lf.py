from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import load_filtered_samples
from sklearn.preprocessing import OneHotEncoder
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import warnings
warnings.filterwarnings("ignore")

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


    # if (filename != ""): plt.savefig("roc_multi_{}.png".format(filename))



def process_feature():
    latent_feature = read_csv('./results/saved_representation.tsv')
    filtered_samples = load_filtered_samples()
    labels = pd.read_csv('./data/samples_label.tsv', sep='\t', index_col=0)
    latent_feature = latent_feature.filter(items=filtered_samples, axis=0)
    labels = labels.filter(items=filtered_samples, axis=0)

    latent_feature = latent_feature.values
    # label = label['label'].values
    label = labels['label'].to_numpy()
    label_recon = np.zeros([len(label)])
    for i,d in enumerate(label):
        c = d
        if d > 3:
            c -= 1
        if d > 19:
            c -= 1
        if d > 22:
            c -= 1
        if d > 27:
            c -= 1
        if d > 30:
            c -= 2
        label_recon[i] = c
    label = label_recon
    X, y = latent_feature, label
    return X, y




def train_classifier(X, y, classifier='svm'):
    print(X.shape)
    print(y.shape)

    ## 5_cross_validation
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

        clf = SVC(gamma='scale', probability=True)

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
    X, y = process_feature()
    _ = train_classifier(X, y, classifier='svm')



