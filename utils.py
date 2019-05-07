from sklearn import metrics
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

def NMI(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def ARI(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


def ACC(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size