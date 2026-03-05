import utils as u
import numpy as np


""" 
# IRIS CONFIG
INPUT_FILE = "lab02/data/iris.csv"
feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
"""

""" 
top=0.999,
bottom=0.08,
left=0.05,
right=0.98,
hspace=0.6,
wspace=0.4

 """


# PROJECT CONFIG
INPUT_FILE = "lab02/data/trainData.txt"


if "__main__":
    D, L, labels = u.load_data(INPUT_FILE)

    feature_names = [f'F{i}' for i in range(0, D.shape[0])]

    means, variances, stds = u.get_class_descriptors(D, L, labels)

    np.set_printoptions(linewidth=400)

    print(labels)
    print("\n")

    print(means)
    print("\n")

    print(variances)
    print("\n")

    print(stds)
    print("\n")

    cov = u.get_cov(D)
    print(D)
  
    # u.display_histograms(D, L, labels=labels, feature_names=feature_names)
    # u.display_feature_pairs(D, L, labels=labels, feature_names=feature_names)