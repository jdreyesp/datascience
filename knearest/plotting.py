#Standard library imports
import math

#Third party imports
from matplotlib import pyplot as plt
from typing import List, Dict

#Local application imports
from scratch.linear_algebra import Vector


def plot(metrics: List[str], n_classes: int, data: Dict[str, List[Vector]]):

    pairs = [(i,j) for i in range(len(metrics)) for j in range(len(metrics)) if i < j]
    assert(n_classes <= 8)
    marks = ['+', '.', 'x', '*', '#', "-", "&", "$"] #we accept up to 8 different classes

    marks = marks[:n_classes]

    cols = 3
    rows = math.ceil(len(pairs) / cols)

    fig, axs = plt.subplots(rows, cols, squeeze=False)

    for row in range(rows):
        for col in range(cols):
            if (cols * row + col) < len(pairs):
                i,j = pairs[cols * row + col]
                axs[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
                axs[row][col].set_xticks([])
                axs[row][col].set_yticks([])

                for mark, (data_labels, data_points) in zip(marks, data.items()):
                    xs = [point[i] for point in data_points]
                    ys = [point[j] for point in data_points]
                    axs[row][col].scatter(xs, ys, marker=mark, label=data_labels)

    axs[-1][-1].legend(loc='lower right', prop={'size': len(pairs)})
    plt.show()


#metrics = ["age", "pregnant", "diabetic"]
#n_classes = 3 #countries
#data = {"Spain": [[35.0, 0.0, 0.0], [36.0, 0.0, 1.0], [40.0, 1.0, 0.0]],
#        "UK": [[40.0, 1.0, 1.0], [26.0, 0.0, 0.0], [55.0, 0.0, 1.0]],
#        "Netherlands": [[65.0, 1.0, 1.0], [20.0, 0.0, 0.0]]}

#plot(metrics, n_classes, data)
