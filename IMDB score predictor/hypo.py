import numpy as np
import math

# y : rating
theta = [1 for i in range(7)]

#1D parameter X
def hypothesis(X):
    global theta
    h = 0.0
    for i in range(len(theta)):
        h += theta[i]* X[i]
    return h
#1D parameter X
def cost_function(X,Y,number_of_records):
    j = 0.0

    for i in range(number_of_records):
        #print(hypothesis(X[i]))
        e = hypothesis(X[i]) - Y[i]

        j += e*e


    j = j* 1.0/number_of_records

    return j
#2D parameter X
def cost_fuction_derivate(X,Y,index,i):
    return (hypothesis(X[index]) - Y[index]) * X[index][i]

def gradient_descent(X,Y,number_of_records):
    global theta
    previous_cost = -100
    alpha = .3
    cost = cost_function(X,Y,number_of_records)
    while abs(cost - previous_cost) > 0.0000000000001:
        previous_cost = cost

        for i in range(len(theta)):
            dj=0
            for j in range(number_of_records):
                dj += cost_fuction_derivate(X,Y,j,i)

            theta[i] -= alpha * dj * (1.0/number_of_records)

        cost = cost_function(X,Y,number_of_records)
        print(cost)

# matX =[[1,2,5,5,2,1],[1,3,3,7,4,6],[1,4,3,2,3,1],[1,5,2,1,3,5],[1,2,3,1,1,1],[1,4,4,9,5,6]]
# matY =[10,11,17,12,18,15]
# theta = [1 for i in range(6)]
# print(cost_function(matX,matY,6))
 #gradient_descent(matX,matY,4)
# print(theta)

# for i in matX:
#     print(hypothesis(i))
