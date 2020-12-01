from typing import Tuple

from scratch.linear_algebra import Vector
from scratch.statistics import correlation, standard_deviation, mean

#Making the predictions
def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha

#Error for each x_i, y_i pair
def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """The error from predicting beta * x_i + alpha when the value is y_i"""
    return predict(alpha, beta, x_i) - y_i

#We calculate the sum of the squared errors to avoid x_1 too high and x_2 too low to cancel each other
def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x,y))

#We get now alpha and beta we need to calculate
def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """Given two vectors x and y, find the least-squares values of alpha and beta"""
    beta = correlation(x,y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

assert least_squares_fit(x, y) == (-5, 3)

from scratch.statistics import num_friends_good, daily_minutes_good

#We expect a user with n friends to spend 22.95 + n * 0.903 minutes on the site each day
alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905


