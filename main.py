import random
import csv


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        header = dataset.pop(0)  # Remove and store the header row 
        for x in range(len(dataset)):
            for y in range(5):  
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


trainingSet = []
testSet = []
loadDataset(r'Iris.csv', 0.66, trainingSet, testSet)
print('Train: ' + repr(len(trainingSet)))
print('Test: ' + repr(len(testSet)))

# SIMILARITY
import math

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow(((float(instance1[x])) - float(instance2[x])), 2)
    return math.sqrt(distance)


data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print('Distance: ' + repr(distance))

# Look for KNN
import operator


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = getNeighbors(trainSet, testInstance, 1)
print(neighbors)


# predict a respons e based on those neighbours:
# you can allow each neighbour to vote for the class attribute and take th e majority vote as a prediction
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


neighbors = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
response = getResponse(neighbors)
print(response)


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)



import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Load the Iris dataset
iris = pd.read_csv('Iris.csv')

# Basic data exploration
iris.groupby('Species').describe() 

# Define color map
color_map = {'Iris-setosa': 'blue', 'Iris-versicolor': 'green', 'Iris-virginica': 'red'}

# Visualize the relationship between two features
plt.scatter(iris['SepalLengthCm'], iris['SepalWidthCm'], c=iris['Species'].map(color_map))
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Model training and hyperparameter tuning
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

accuracy_scores = []
for k in range(1, 11):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  accuracy_scores.append(knn.score(X_test, y_test))

plt.plot(range(1, 11), accuracy_scores)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.show()


# combine all using one main function:
def main():
    # Prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('Iris.csv', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))

    # Model training and hyperparameter tuning (no changes needed here)
    X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = iris['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Generate predictions (using scikit-learn)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)  
    predictions = knn.predict(X_test)

    # Calculate Accuracy
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(str(accuracy)) + '%')

main()

