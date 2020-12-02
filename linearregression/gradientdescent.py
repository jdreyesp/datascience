import random
import tqdm
from scratch.gradient_descent import gradient_step
from scratch.statistics import num_friends_good, daily_minutes_good
from scratch.linear_algebra import Vector

from typing import List

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

def gradient_descent(x: List[float] ,y: List[float]) -> float:
    num_epochs = 10000
    random.seed(0)

    guess = [random.random(), random.random()] #choose random value to start

    learning_rate = 0.00001

    with tqdm.trange(num_epochs) as t:
        for _ in t:
            alpha, beta = guess

            #Partial derivative of loss with respect to alpha
            grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                         for x_i, y_i in zip(x,
                                             y))

            #Partial derivative of loss with respect to beta
            grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                         for x_i, y_i in zip(x,
                                             y))

            #The loss is the error in our predicted value of m and c. The goal is to minimize this error to obtain the most accurate value of alpha,beta
            loss = sum_of_sqerrors(alpha, beta, x, y)
            t.set_description(f"loss: {loss:.3f}")

            #Finally, update the guess
            guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)

    return guess

#We expect a user with n friends to spend 22.95 + n * 0.903 minutes on the site each day
#alpha, beta = gradient_descent(num_friends_good, daily_minutes_good)
#print (alpha, beta)
#assert 22.9 < alpha < 23.0
#assert 0.9 < beta < 0.905