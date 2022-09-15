import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt


def load_dataset(filename, split):
    df = pd.read_csv(filename, header=None)
    array = df.to_numpy()
    random.shuffle(array)
    training_len = int(len(array)*split)
    training_set = array[:training_len]
    test_set = array[training_len:]
    return training_set, test_set


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def knn(split, k):
    url = 'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data'
    training_set, test_set = load_dataset(url, split)
    predictions = []
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[x], k)
        result = get_response(neighbors)
        predictions.append(result)
    accuracy = get_accuracy(test_set, predictions)
    return accuracy


def main():
    plt.style.use('seaborn-whitegrid')

    # Collect accuracies for k values 1-20.
    accuracies = []
    for x in range(20):
        k_accuracy = []
        for y in range(5):
            k_accuracy.append(knn(0.67, x + 1))
            plt.plot(x + 1, k_accuracy[-1], 'bo')
        accuracies.append(k_accuracy)
        k_average = 0
        for accuracy in k_accuracy:
            k_average += accuracy
        k_average /= len(k_accuracy)
        plt.text(x + 1.1, max(k_accuracy) + 2, repr(k_average) + '%', size='x-small', weight='bold')

    # Calculate average accuracy.
    total_items = 0
    average_accuracy = 0
    for x in accuracies:
        for y in x:
            total_items += 1
            average_accuracy += y

    average_accuracy /= total_items

    # Build graph.
    plt.ylim(0, 100)
    plt.ylabel('Accuracy\nNoted % values are per-k averages')
    plt.xlim(0, len(accuracies))
    plt.xticks(range(20))
    plt.xlabel('k Value')
    plt.title('KNN Accuracies for Increasing k-Values\nTotal Average accuracy rating is: ' + repr(average_accuracy) + '%', weight='bold')
    plt.show()

    print('Average accuracy is ' + repr(average_accuracy) + '%')


main()
