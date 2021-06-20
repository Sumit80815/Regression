import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./Training Data/Train.csv")
one_arr = np.ones((df.shape[0],))
x = np.c_[one_arr, df['feature_1'], df['feature_2'], df['feature_3'],  df['feature_4'], df['feature_5']]
y = df['target'].values
y = y.reshape((-1,))
for i in range(1,6):
    x[:,i] = (x[:,i] - x[:,i].mean())/x[:,i].std()
def hypothesis(x,theta):
    return np.dot(x,theta)
def error(x,theta,y):
    err = 0.0
    m = x.shape[0]

    for i in range(m):
        hx = hypothesis(x[i],theta)
        err += (hx-y[i])**2

    return err
def gradient(x,theta,y):
    m = x.shape[0]

    grad = np.zeros((theta.shape))

    for i in range(m):
        hx = hypothesis(x[i],theta)

        grad += (hx - y[i])*x[i]

    return grad/m
def gradient_descent(x,y,learning_rate = 0.01):


    theta = np.zeros((x.shape[1],))

    err_list = []
    theta_list = []
    

    for i in range(1000):
        grad = gradient(x,theta,y)
        err = error(x,theta,y)

        err_list.append(err)
        theta_list.append(theta)


        theta -= (learning_rate*(grad))


    return theta,err_list,theta_list
final_theta , err_list , theta_list = gradient_descent(x,y)
plt.plot(err_list)
print(final_theta)
