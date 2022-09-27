import random

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.cluster import KElbowVisualizer

matplotlib.use('TkAgg')

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


# TODO determine the best k for k-means

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 8))
visualizer.fit(X)
visualizer.show()

# TODO calculate accuracy for best K
rand = random.randint(0, 101)
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.4, random_state=rand)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
accuracy_score(X_test, predicted)

# TODO draw a confusion matrix
