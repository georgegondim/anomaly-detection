import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss, roc_auc_score, roc_curve


def plot_roc(y, weight_dict):

    models = ['mlp', 'logistic', 'svm']
    for dataset in y:
        if weight_dict is not None:
            weights = np.empty_like(y[dataset])
            weights[y[dataset] == 0] = weight_dict[0]
            weights[y[dataset] == 1] = weight_dict[1]
        else:
            weights = None
        plt.figure()
        for model in models:
            score_file = '../outputs/' + model + '_' + dataset + '.npy'
            y_score = np.load(score_file)
            fpr, tpr, _ = roc_curve(y[dataset], y_score, sample_weight=weights)
            roc_auc = roc_auc_score(y[dataset], y_score, sample_weight=weights)
            loss = log_loss(y[dataset], y_score, 10e-7, sample_weight=weights)

            plt.plot(fpr, tpr, label=model + ': AUC = %0.5f' % roc_auc)
            plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.savefig('../imgs/' + dataset + '.png')
            print('%s,%s,%f,%f' % (dataset, model, roc_auc, loss))


if __name__ == '__main__':
    ################################################################
    # Data
    data = np.load("../data/creditcard_train.npz")
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    input_dim = x_train.shape[1]
    df = pd.read_csv('../data/creditcard_test.csv')
    data = df.drop(['Unnamed: 0', 'Time'], 1).as_matrix()
    x_test, y_test = data[:, :-1], data[:, -1]
    ###############################################################

    weight_dict = {
        0: len(y_train) / np.sum(1 - y_train),
        1: len(y_train) / np.sum(y_train)
    }
    y = {}
    y['train'] = y_train
    y['val'] = y_val
    y['test'] = y_test
    plot_roc(y, weight_dict)
