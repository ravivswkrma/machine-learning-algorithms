# Import standard libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from numpy.linalg import eig
import math
import random

df = pd.read_csv('A2Q2Data_train.csv',header = None) #Reading dataset
data_matrix = np.array(df) #Storing in matrix


x = data_matrix[:,:100]  #Storing 1st 100 columns in the x
y = data_matrix[:,100:101] #Storing the last column in y which is the label
x_transpose = x.T
x_xt = x_transpose.dot(x)
x_y = x_transpose.dot(y)
xxt_inverse = np.linalg.inv(x_xt)
Wml = xxt_inverse.dot(x_y) #Calculating the Wml


def objective_func(w): # This function computes the L2 norm of (w-Wml)
    v = w - Wml
    vt = v.T
    return math.sqrt(vt.dot(v))
    
    
def gradient(w): #Computes 2((XX^t)W-XY)
    temp = x_xt.dot(w)
    temp2 = temp-x_y
    return 2*temp2
    

w = np.zeros((100,1)) #Initializing w with 0
ans_list = []
obj_value = objective_func(w)
ans_list.append(obj_value)


t=1
for t in range(1,1000): #In every iteration we are computing the w and appending the correspondig value of objective function to ans list
    ita = 0.000000001/t
    w = w - ita*gradient(w)
    ans_list.append(objective_func(w))
    
# print(ans_list)

plt.plot(range(1000),ans_list) #Plotting the result
plt.xlabel('Number of Iterations     x-axis')
plt.ylabel('Mean Squared Error    y-axis')
plt.title('Cost vs Iterations Analysis')
