from matplotlib import pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
from scipy.stats import mode
import seaborn as sns
sns.set()

# I need this statement for some reason.
matplotlib.use('TkAgg')

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

# Determine and visualize the best K for the data set.
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 8))
visualizer.fit(X)
visualizer.show()

# Plot the data set using the aforementioned calculated best K.
kmeans = KMeans(4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# Calculate the accuracy for the calculated best K.
# Mode throws a "FutureWarning" but it's not deprecated yet :)
clusters = kmeans.fit_predict(X)
labels = np.zeros_like(clusters)
for i in range(visualizer.elbow_value_):
    mask = (clusters == i)
    labels[mask] = mode(y_true[mask])[0]
print(accuracy_score(y_true, labels))

# Draw and display confusion matrix.
matrix = confusion_matrix(y_true, labels)
sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
