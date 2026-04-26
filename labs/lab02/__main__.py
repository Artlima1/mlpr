import numpy as np
from utils.utils import load_data, get_class_descriptors, get_cov, plot_feature_pairs, plot_histograms


""" 
# IRIS CONFIG
INPUT_FILE = "lab02/data/iris.csv"
feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
"""


# PROJECT CONFIG
INPUT_FILE = "data/trainData.txt"


if "__main__":
    D, L, labels = load_data(INPUT_FILE)

    feature_names = [f'F{i}' for i in range(0, D.shape[0])]

    means, variances, stds = get_class_descriptors(D, L, labels)

    np.set_printoptions(linewidth=400)

    print(labels)
    print("\n")

    print(means)
    print("\n")

    print(variances)
    print("\n")

    print(stds)
    print("\n")

    cov = get_cov(D)
    print(D)
  
    # plot_histograms(D, L, labels=labels, feature_names=feature_names)
    # plot_feature_pairs(D, L, labels=labels, feature_names=feature_names)