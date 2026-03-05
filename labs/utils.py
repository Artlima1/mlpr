
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path) -> Tuple[np.ndarray, np.ndarray, List]:
    """Load data from a file."""
    D = []
    L = []
    labels = set()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            D.append([float(x) for x in parts[:-1]])
            L.append(parts[-1])
            labels.add(parts[-1])
    D = np.array(D).T
    L = np.array(L).T.reshape(1,len(L))
    return D, L, list(labels)

def get_mean(D: np.ndarray) -> np.ndarray:
    """Calculate the mean of each column of D."""
    return D.mean(1).reshape(D.shape[0], 1)


def get_variance(D: np.ndarray) -> np.ndarray:
    return D.var(1).reshape(D.shape[0], 1)

def get_std(D: np.ndarray) -> np.ndarray:
    return D.std(1).reshape(D.shape[0], 1)

def get_cov(D: np.ndarray) -> np.ndarray:
    """Calculate the covariance matrix of D."""
    mean = get_mean(D)
    return (D - mean) @ (D - mean).T / float(D.shape[1])

def get_class_descriptors(D: np.ndarray, L: np.ndarray, labels: List):
    means = []
    variances = []
    stds = []

    for l in labels:
        samples = L[0] == l
        data = D[:,samples]

        means.append(get_mean(data)[:,0])
        variances.append(get_variance(data)[:,0])
        stds.append(get_std(data)[:,0])

    means = np.array(means).T
    variances = np.array(variances).T
    stds = np.array(stds).T

    return (means, variances, stds)


def display_histograms(D: np.ndarray, L: np.ndarray, labels: List, feature_names: List = []) -> None:
    n_features = D.shape[0]
    if len(feature_names) == 0:
        feature_names = [f'F{i}' for i in range(0,n_features)] 
    
    plt.figure()
    for i in range (0, n_features):
        plt.subplot(1, n_features, i+1)
        for l in labels:
            samples = L[0] == l
            data = D[i,samples]
            plt.hist(data, density=True, bins=10, histtype="barstacked", label=l, alpha=0.6)
            plt.legend()
            plt.xlabel(feature_names[i])
            plt.ylabel("density")
    
    plt.show()

def display_feature_pairs(D: np.ndarray, L: np.ndarray, labels: List, feature_names: List=[]) -> None:    
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
                    samples = L[0] == l
                    data = D[i_f1,samples]
                    plt.hist(data, density=True, bins=10, histtype="barstacked", label=l, alpha=0.6)
                    plt.ylabel("density")
            else:
                for l in labels:
                    samples = L[0] == l
                    x = D[i_f1,samples]
                    y = D[i_f2,samples]
                    plt.scatter(x, y, label=l, s=1)
                    plt.ylabel(feature_names[i_f2])
            
            plt.xlabel(feature_names[i_f1])
            plt.legend(fontsize=5)

    # plt.tight_layout()
    plt.show()