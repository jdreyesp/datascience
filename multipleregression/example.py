from typing import List
import random

from multipleregression.gradientdescent import least_squares_fit

random.seed(0)

#Multiple regression applies the same logic as simple regression, but this time 2...N independent variables are involved, working with this model instead:
# y_i = alpha + beta1 *x_1... + betaN * x_n + epsilon

#In multiple regression the vector of parameters is usually called beta, and that will be:
# beta = [alpha, beta_1 ... beta_k]
# and the input data from the existing dataset we will work with will be represented as:
# x_i = [1, x_1, ..., x_ik]

#Let's say I have this data set, that represents average masses for women as a function of their heights and pregnancy:
#Height (m), x1	1.47	1.50	1.52	1.55	1.57	1.60	1.63	1.65	1.68	1.70	1.73	1.75	1.78	1.80	1.83 <- INDEPENDENT VARIABLE
#Prgn (y/n), x2 0       0       0       1       0       0       1       1       0       0       1       0       0       1       1    <- INDEPENDENT VARIABLE
#Mass (kg), yi	52.21	53.12	54.48	55.84	57.20	58.57	59.93	61.29	63.11	64.47	66.28	68.10	69.92	72.19	74.46 <- DEPENDENT VARIABLE
heights: List[float] = [1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83]
pregnant: List[float] = [0,0,0,1,0,0,1,1,0,0,1,0,0,1,1]
masses: List[float] = [52.21, 53.12, 54.48, 58.84, 57.20, 58.57, 61.93, 61.29, 63.11, 64.47, 67.50, 68.10, 69.92, 76.19, 78.15]


#Using gradient-descent approach, we can calculate betas:
input = [[x_1, x_2, x_3] for x_1, x_2, x_3 in zip([1 for _ in heights], heights, pregnant)]
betas = least_squares_fit(input, masses, 0.001, 5000, 25)

#All-else-being-equal estimation
print(f"Constant: {betas[0]})")
print(f"Height: {betas[1]})") #Each additional meter corresponds to 27 more kgs / meter => Each additional cm corresponds to 200 more grams.
print(f"Pregnant: {betas[2]})") #Being pregnant corresponds on having 4-5 more kilos average.

#This model doesn't capture the interactions between variables
#i.e. it's possible that the effect of being pregnant is different for short women and tall women

#We can predict new Masses based on incoming new Heights and pregnancy. So if, based on the previous dataset, I receive:
# height = 1,72
# pregnant = 1
# This function will be applied:
# mass = 7.350997075238178 + (4.848487626011433 * 1,72) + (4.312706130154798 * 1) =
alpha = betas[0]
beta1 = betas[1]
beta2 = betas[2]
new_height = 1.72
pregnant = 1
new_mass = alpha + (beta1 * new_height) + (beta2 * pregnant)

print(f"Multiple regression with Gradient descent for height: {new_height} and pregnant: {bool(pregnant)}: {new_mass}")