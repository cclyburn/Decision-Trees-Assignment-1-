import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors

"""
utility functions
"""

def entropy(S):
    amountOfElements = len(S)
    occurance = {}
    for element in S:
        occurance[element] = occurance.get(element, 0) + 1

    probabilites = {k: v / amountOfElements for k, v in occurance.items()}

    entropy_val = 0
    for prob in probabilites.values():
        entropy_val -= prob * math.log2(prob)
    return float(entropy_val)


def information_gain(S, S1, S2):
    entropyS = entropy(S)
    entropyS1 = entropy(S1)
    entropyS2 = entropy(S2)
    S1division = len(S1) / len(S)
    S2division = len(S2) / len(S)
    childrenEntropy = (S1division * entropyS1) + (S2division * entropyS2)
    informationGain = entropyS - childrenEntropy
    return float(informationGain)


def best_split_for_feature(x, y):
    sorted_indices = np.argsort(x)
    sortedX = x[sorted_indices]
    sortedY = y[sorted_indices]
    midpoints = [(sortedX[i] + sortedX[i+1]) / 2 for i in range(len(sortedX)-1)]

    best_informationGain = -1
    best_split = None

    for split in midpoints:
        S1 = sortedY[sortedX < split]
        S2 = sortedY[sortedX >= split]
        info_gain = information_gain(sortedY, S1, S2)
        if info_gain > best_informationGain:
            best_informationGain = info_gain
            best_split = split

    return float(best_split), float(best_informationGain)


def best_feature_to_split_on(X, y):
    best_feature = -1
    best_info_gain = -float('inf')
    best_split_value = None

    for feature_index in range(X.shape[1]):
        split_value, info_gain = best_split_for_feature(X[:, feature_index], y)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature_index
            best_split_value = split_value
        elif info_gain == best_info_gain and (best_feature == -1 or feature_index < best_feature):
            best_feature = feature_index
            best_split_value = split_value

    return best_feature, float(np.round(best_info_gain, 3)), float(np.round(best_split_value, 3))


def load_random_dataset():
    X = np.load('../datasets/X_random.npy')
    y = np.load('../datasets/y_random.npy')
    return X, y


def load_circle_dataset():
    X_train = np.load('../datasets/X_circle_train.npy')
    y_train = np.load('../datasets/y_circle_train.npy')
    X_val = np.load('../datasets/X_circle_val.npy')
    y_val = np.load('../datasets/y_circle_val.npy')
    return X_train, y_train, X_val, y_val


def plot_decision_boundary(X, y, model, title, resolution=0.02):
    fig, ax = plt.subplots()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap = plt.cm.RdYlBu
    norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, np.max(y) + 1.5), ncolors=cmap.N)

    ax.contourf(xx, yy, Z, alpha=0.75, cmap=cmap)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', norm=norm, cmap=cmap, s=50)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_title(title)

    unique_classes = np.unique(y)
    class_labels = [f'Class {cls}' for cls in unique_classes]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(cls)), markersize=10) for cls in unique_classes]
    ax.legend(handles, class_labels, loc='best')

    fig.show()  # <-- interactive display
    return fig


def plot_dataset(X, y, title, xlabel='Feature 0', ylabel='Feature 1', legend=True):
    fig, ax = plt.subplots()
    cmap = plt.cm.RdYlBu
    norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, np.max(y) + 1.5), ncolors=cmap.N)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', norm=norm, cmap=cmap, s=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    unique_classes = np.unique(y)
    class_labels = [f'Class {cls}' for cls in unique_classes]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(cls)), markersize=10) for cls in unique_classes]
    if legend:
        ax.legend(handles, class_labels, loc='best')

    fig.show()  
    return fig
