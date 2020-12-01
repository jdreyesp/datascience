from voting import majority_vote
from typing import NamedTuple, List
from scratch.linear_algebra import Vector, distance

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int,
                 labeled_points: List[LabeledPoint],
                 new_point: Vector) -> str:
    #Order the labeled points from nearest to farthest.
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))

    #Find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    #and let them vote
    return majority_vote(k_nearest_labels)