# Import standard libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from numpy.linalg import eig
import math

df = pd.read_csv('A2Q2Data_train.csv',header = None)
data_matrix = np.array(df)

test_df = pd.read_csv('A2Q2Data_test.csv',header = None)
test_data_matrix = np.array(test_df)


x_test = test_data_matrix[:,:100]
y_test = test_data_matrix[:,100:101]


x = data_matrix[:,:100]
y = data_matrix[:,100:101]
x_transpose = x.T
x_xt = x_transpose.dot(x)
x_y = x_transpose.dot(y)
xxt_inverse = np.linalg.inv(x_xt)
Wml = xxt_inverse.dot(x_y)


w = np.zeros((100,1))
ans_list = []
Wr = np.zeros((100,1))

def objective_func(w,l):
    temp = x.dot(w)
    temp1 = temp - y 
    mse = np.square(temp1).mean()
    ridge = mse + l*(np.linalg.norm(w)*np.linalg.norm(w))
    return ridge

def error_calculation_ridge(w,x_test,y_test,l):
    temp = x_test.dot(w)
    temp1 = temp - y_test 
    mse = np.square(temp1).mean()
    ridge_mse = mse + l*(np.linalg.norm(w)*np.linalg.norm(w))
    return ridge_mse


def error_calculation_Wml(Wml,x_test,y_test):
    y_predicted = np.matmul(x_test,Wml)
    mse = np.square(np.subtract(y_test,y_predicted)).mean()
    return mse

    
def gradient(w,l):
    temp = x_xt.dot(w)
    temp1 = temp - x_y
    temp2 = temp1 + l*w
    return 2*temp2
    

def ridge_gradient(w,Wr,l):        
    t=1
    for t in range(1,1000):
        ita = 0.000001
        w = w - ita*gradient(w,l)   
    ans_list.append(objective_func(w,l)) 
    return w

l=2
i=0
temp_wr = np.zeros((100,1))
arr = np.zeros((10,1))
while(l<100):
    arr[i]=l
    i=i+1
    temp_wr = ridge_gradient(w,Wr,l)
    if l == 2:
        Wr = temp_wr
        ans_lambda = l
    l = l+10
    
    

print("Minimum error at Lambda = ",ans_lambda)
print("Test error for best Wr = ",error_calculation_ridge(Wr,x_test,y_test,0))
print("Test error for Wml = ",error_calculation_Wml(Wml,x_test,y_test))



plt.plot(arr,ans_list)
plt.xlabel('Lambda        x-axis')
plt.ylabel('Error function      y-axis')
plt.title('Error in the validation set as a function of lambda')