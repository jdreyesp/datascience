from typing import List

import random
import tqdm

from scratch.linear_algebra import dot, Vector, vector_mean
from scratch.gradient_descent import gradient_step

from scratch.statistics import daily_minutes_good

def predict(x: Vector, beta: Vector) -> float:
    """
    :param x: vector of [1, x_1... x_n] where x_1..n is the existing input data values
    :param beta: vector of [alpha, beta_1... beta_n] where beta_1..n is a mutually exclusive variables representing the independent values
    in a multiple regression.
    :return:
    """
    return dot(x, beta)

def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y

def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2

# x = [1,2,3]
# y = 30
# beta = [4, 4, 4] #so prediction = 4 + 8 + 12 = 24

# assert error(x, y, beta) == -6
# assert squared_error(x, y, beta) == 36

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]

# assert sqerror_gradient(x, y, beta) == [-12, -24, -36]

def least_squares_fit(xs: List[Vector],
                      ys: List[float],
                      learning_rate: float = 0.001,
                      num_steps: int = 1000,
                      batch_size: int = 1) -> Vector:
    """Find the beta that minimizes the sum of squared errors assuming the model y = dot(x, beta)"""

    #Starts with a random guess
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess) for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

