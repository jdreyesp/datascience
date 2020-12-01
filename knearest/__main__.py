import requests
import random
import csv

from typing import List, Tuple, Dict
from collections import defaultdict

from scratch.linear_algebra import Vector
from scratch.machine_learning import split_data
from knearest import knn_classify, LabeledPoint
from plotting import plot

def main():
    #RETRIEVE DATA

    data = requests.get(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    )

    with open('iris.dat', 'w') as f:
        f.write(data.text)

    def parse_iris_row(row: List[str]) -> LabeledPoint:
        """
        sepal_length, sepal_width, petal_length, petal_width, class
        """
        measurements = [float(value) for value in row[:-1]]
        label = row[-1].split("-")[-1]

        return LabeledPoint(measurements, label)

    with open('iris.dat', 'r') as f:
        reader = csv.reader(f)
        iris_data = [parse_iris_row(row) for row in reader if row != []]

    #K NEAREST PREDICTION
    random.seed(12)
    iris_train, iris_test = split_data(iris_data, 0.70)

    confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    num_correct = 0

    for iris in iris_test:
        predicted = knn_classify(5, iris_train, iris.point)
        actual = iris.label

        if predicted == actual:
            num_correct += 1

        confusion_matrix[(predicted, actual)] += 1

    pct_correct = num_correct / len(iris_test)
    print(pct_correct, confusion_matrix)

    #PLOT

    points_by_species: Dict[str, List[Vector]] = defaultdict(list)
    for iris in iris_data:
        points_by_species[iris.label].append(iris.point)

    plot(["sepal_length", "sepal_width", "petal_lenght", "petal_width"], 3, points_by_species)

if __name__ == '__main__':
    main()