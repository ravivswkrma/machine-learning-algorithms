# Import standard libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from numpy.linalg import eig
import math
import random

df = pd.read_csv('A2Q2Data_train.csv',header = None)
data_matrix = np.array(df)

w = np.zeros((100,1))
ans_list = []

def objective_func(w,Wml):
    v = w - Wml
    vt = v.T
    return math.sqrt(vt.dot(v))
    
    
def gradient(w,Wml,x_xt,x_y):
    temp = x_xt.dot(w)
    temp2 = temp-x_y
    return 2*temp2
    

def stochastic_grad(Wml,w,i,x_xt,x_y):        
    t=1
    for t in range(1,1000):
        ita = 0.0001
        w = w - ita*gradient(w,Wml,x_xt,x_y)
    ans_list.append(objective_func(w,Wml))        
               


def analyticalWml():
    x = data_matrix[:,:100]
    y = data_matrix[:,100:101]
    x_transpose = x.T
    x_xt = x_transpose.dot(x)
    x_y = x_transpose.dot(y)
    xxt_inverse = np.linalg.inv(x_xt)
    Wml = xxt_inverse.dot(x_y)
    return Wml

i=1
Wml = analyticalWml()
obj_value = objective_func(w,Wml)
ans_list.append(obj_value)

while(i<=10):
    np.random.shuffle(data_matrix)
    x = data_matrix[:100,:100]
    y = data_matrix[:100,100:101]
    x_transpose = x.T
    x_xt = x_transpose.dot(x)
    x_y = x_transpose.dot(y)
    xxt_inverse = np.linalg.inv(x_xt)
    i = i+1
    stochastic_grad(Wml,w,i,x_xt,x_y)
    

plt.plot(ans_list)
plt.xlabel('Number of Iterations  x-axis')
plt.ylabel('Mean Squared Error     y-axis')
plt.title('Cost vs Iterations Analysis')


