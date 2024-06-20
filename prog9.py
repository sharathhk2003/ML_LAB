import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

data = load_iris().data[:6]

def proximity_matrix(data):
    return np.linalg.norm(data[:, np.newaxis] - data, axis=2)

def plot_dendrogram(data, method):
    dendrogram(linkage(data, method=method))
    plt.title(f'Dendrogram - {method.capitalize()} Linkage')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

print("Proximity matrix:\n", proximity_matrix(data))
plot_dendrogram(data, 'single')
plot_dendrogram(data, 'complete')
