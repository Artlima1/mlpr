import numpy as np
from utils import get_cov, perform_PCA, load_data, display_feature_pairs, get_class_covariances, perform_LDA



# IRIS CONFIG
INPUT_FILE = "data/iris.csv"



# PROJECT CONFIG
# INPUT_FILE = "data/trainData.txt"

m = 2

if "__main__":
    D, L, labels = load_data(INPUT_FILE)

    np.set_printoptions(linewidth=400)

    # Covariance
    # C = get_cov(D)
    # print("Covariance Matrix: ")
    # print(C)

    # PCA
    Dp = perform_PCA(D, m)
    feature_names = [f'PC{i}' for i in range(1, m+1)]
    display_feature_pairs(Dp, L, labels, feature_names)

    # LDA
    # Sb, Sw = get_class_covariances(D, L, labels)
    # print("Between class covariance Matrix (SB): ")
    # print(Sb)
    # print("Within Class Covariance Matrix (SW): ")
    # print(Sw)

    # Dp = perform_LDA(D, L, labels, m)
    # feature_names = [f'PC{i}' for i in range(1, m+1)]
    # display_feature_pairs(Dp, L, labels, feature_names)



