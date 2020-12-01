import random
import tqdm
from scratch.gradient_descent import gradient_step
from scratch.statistics import num_friends_good, daily_minutes_good
from linearregression.linearreg import error, sum_of_sqerrors

from typing import List

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