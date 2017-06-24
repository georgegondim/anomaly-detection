import joblib
from joblib import Parallel, delayed

import numpy as np

import time

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.svm import SVC


def train_model(dict_log):
    weight_dict = {
        0: len(y_train) / np.sum(1 - y_train),
        1: len(y_train) / np.sum(y_train)
    }

    svm = SVC(
        C=dict_log["C"],
        gamma=dict_log["gamma"],
        kernel="rbf",
        probability=True,
        shrinking=False,
        verbose=False,
        cache_size=6000,
        class_weight=weight_dict)

    print("Starting %s" % str(dict_log))
    tic = time.time()
    svm.fit(x_train, y_train)
    probs = svm.predict_proba(x_val)[:, 1]
    dict_log["loss"] = log_loss(y_val, probs)
    dict_log["auc"] = roc_auc_score(y_val, probs)
    dict_log["elapsed"] = time.time() - tic
    print("\tFinished %s" % str(dict_log))
    fp.write(str(dict_log) + "\n")
    fp.flush()
    joblib.dump(
        svm,
        "../svm_model/model.auc:%.6f.loss:%.6f.C:%.4f.gamma:%.4f.pkl" %
        (dict_log["auc"], dict_log["loss"], dict_log["C"], dict_log["gamma"]))

    return dict_log


#  Load Data
data = np.load("../data/creditcard_train.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]

# sort arrays
idx = np.argsort(y_train)
x_train, y_train = x_train[idx], y_train[idx]
idx = np.argsort(y_val)
x_val, y_val = x_val[idx], y_val[idx]

# Parameters to cross-validate
list_C = np.float_power(2, np.arange(-10, 11))[::-1]
list_gamma = np.float_power(2, np.arange(-10, -1))

num_workers = 4
dicts = []
for i in range(len(list_C)):
    for j in range(len(list_gamma)):
        dict_log = {}
        dict_log["C"] = list_C[i]
        dict_log["gamma"] = list_gamma[j]
        dicts.append(dict_log)
print('Total number of experiments: %d' % len(dicts))
print(str(list_C))
print(str(list_gamma))

#fp = open("results.log", "a")

#result = Parallel(n_jobs=num_workers)(
#    delayed(train_model)(dict) for dict in dicts)
