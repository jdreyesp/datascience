from typing import List
from linearregression.linearreg import least_squares_fit
from linearregression.gradientdescent import gradient_descent

#Finding the ^alpha and the ^beta using the previous algorithm and based on an existing dataset (that has an independent variable
# and a dependent variable) that generates a linear regression function that will predict the dependent variable values from the
# independent values

#Let's say I have this data set, that represents average masses for women as a function of their height:
#Height (m), xi	1.47	1.50	1.52	1.55	1.57	1.60	1.63	1.65	1.68	1.70	1.73	1.75	1.78	1.80	1.83 <- INDEPENDENT VARIABLE
#Mass (kg), yi	52.21	53.12	54.48	55.84	57.20	58.57	59.93	61.29	63.11	64.47	66.28	68.10	69.92	72.19	74.46 <- DEPENDENT VARIABLE
heights: List[float] = [1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83]
masses: List[float] = [52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46]

#Applying linearregresion using ordinary least squares (OLS) (linearreg.py) we find out that:
# (alpha,beta) = (61.6746, -39.7468)
alpha, beta = least_squares_fit(heights, masses)
print (alpha, beta)

#Based on the model function:
# y = alpha + beta * x
#(which describes a line with slope beta and intercepts alpha in its y axis)

#We can predict new Masses based on incoming new Heights. So if, based on the previous dataset, I receive:
# height = 1,72
# This function will be applied:
# mass = 61,6746 + (-39,7468 * 1,72) =
new_height = 1.72
new_mass = alpha + (beta * new_height)
print(f"Linear regression with OLS: {new_mass}")


#Same applies for gradient descent approach
alpha, beta = gradient_descent(heights, masses)
print (alpha, beta)
new_height = 1.72
new_mass = alpha + (beta * new_height)
print(f"Linear regression with Gradient Descent: {new_mass}")
