
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import scipy


def load_data(file_path) -> Tuple[np.ndarray, np.ndarray, List]:
    """Load data from a file."""
    D = []
    L = []
    labels = set()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            D.append([float(x.strip()) for x in parts[:-1]])
            L.append(parts[-1].strip())
            labels.add(parts[-1].strip())
    D = np.array(D).T
    L = np.array(L).T
    return D, L, list(labels)

def vcol(v: np.ndarray) -> np.ndarray:
    return v.reshape(v.shape[0], 1)

def vrow(v: np.ndarray) -> np.ndarray:
    return v.reshape(1, v.shape[0])

def get_cov(D: np.ndarray) -> np.ndarray:
    """Calculate the covariance matrix of D."""
    mean = vcol(D.mean(1))
    return (D - mean) @ (D - mean).T / float(D.shape[1])

def get_class_descriptors(D: np.ndarray, L: np.ndarray, labels: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = []
    variances = []
    stds = []

    for l in labels:
        data = D[:,L == l]

        means.append(data.mean(1))
        variances.append(data.var(1))
        stds.append(data.std(1))

    means = np.array(means).T
    variances = np.array(variances).T
    stds = np.array(stds).T

    return (means, variances, stds)

def get_PCs(D: np.ndarray, m: int) -> np.ndarray:
    C = get_cov(D)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return P

def get_class_covariances(D: np.ndarray, L: np.ndarray, labels: List) -> Tuple[np.ndarray, np.ndarray]:
    Sw = np.zeros((D.shape[0], D.shape[0]))
    Sb = np.zeros((D.shape[0], D.shape[0]))

    global_mean = vcol(D.mean(1))

    for l in labels:
        data = D[:, L==l]
        n_samples = data.shape[1]

        class_mean = vcol(data.mean(1))
        Sb = Sb + n_samples * ((class_mean-global_mean) @ (class_mean-global_mean).T)

        cov = get_cov(data)
        Sw = Sw + n_samples * cov

    Sw = Sw / D.shape[1]
    Sb = Sb / D.shape[1]

    return (Sb, Sw)

def get_LDs(D: np.ndarray, L: np.ndarray, labels: List, m: int):
    Sb, Sw = get_class_covariances(D, L, labels)

    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]

    return W

def plot_feature_histograms(D: np.ndarray, L: np.ndarray, labels: List, feature_names: List = []) -> None:
    n_features = D.shape[0]
    if len(feature_names) == 0:
        feature_names = [f'F{i}' for i in range(0,n_features)] 
    
    plt.figure(figsize=(15, 3*n_features))
    for i in range (0, n_features):
        plt.subplot(n_features, 1, i+1)
        for l in labels: 
            data = D[i, L==l]
            plt.hist(data, density=True, bins=10, histtype="barstacked", label=l, alpha=0.6)
            plt.legend()
            plt.xlabel(feature_names[i])
            plt.ylabel("density")
    
    plt.show()

def plot_feature_pairs(D: np.ndarray, L: np.ndarray, labels: List, feature_names: List=[]) -> None:    
    n_features = D.shape[0]
    if len(feature_names) == 0:
        feature_names = [f'F{i}' for i in range(0,n_features)] 
    
    plt.figure()
    for i_f1 in range(0, n_features):
        for i_f2 in range(0, n_features):
            chart_index = (i_f1*D.shape[0]) + i_f2 + 1
            plt.subplot(D.shape[0], D.shape[0], chart_index)

            if i_f1==i_f2:
                for l in labels:
                    data = D[i_f1, L==l]
                    plt.hist(data, density=True, bins=10, histtype="barstacked", label=l, alpha=0.6)
                    plt.ylabel("density")
            else:
                for l in labels:
                    x = D[i_f1, L==l]
                    y = D[i_f2, L==l]
                    plt.scatter(x, y, label=l, s=1)
                    plt.ylabel(feature_names[i_f2])
            
            plt.xlabel(feature_names[i_f1])
            plt.legend(fontsize=5)

    # plt.tight_layout()
    plt.show()

def split_db_2to1(D, L, seed=0):

    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def binary_classfication_avgmean(DTR, LTR, DVAL, LVAL, labelA, labelB):
    l1 = max(labelA, labelB)
    l2 = min(labelA, labelB)

    threshold = (DTR[0, LTR==l1].mean() + DTR[0, LTR==l2].mean()) / 2.0

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL[0] >= threshold] = l1
    PVAL[DVAL[0] < threshold] = l2

    misses = (PVAL!= LVAL)
    error_count = misses[(misses==True)].size
    
    return (error_count, misses, threshold)

