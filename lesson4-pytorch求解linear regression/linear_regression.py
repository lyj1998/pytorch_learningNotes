'''
y_true = 1.477*x+0.089+a(高斯分布的噪声)
y_pred = w*x+b
loss = sum((y_pred-y_true)**2)
梯度下降求解w,b
w' = w - lr*∂loss/∂w
b' = b - lr*∂loss/∂b
'''
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
# 平均误差计算函数
# loss = sum((y_pred-y_true)**2)/N
def compute_error_line_regression(w,b,points):
    totalloss = 0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        y_pred = w*x+b
        totalloss+=(y_pred-y)**2
    return totalloss/float(len(points))

# 梯度下降求解函数
def step_gradient(w_current,b_current,points,learningRate):
    """
    w' = w - lr*∂loss/∂w
    b' = b - lr*∂loss/∂b
    loss = ((w*x+b)-y)**2
code:
    gradient_w = 2*((w*x+b)-y)*x
    gradient_b = 2*((w*x+b)-y)
    w = w-learningrate*gradient_w
    b = b-learningrate*gradient_b
    """
#     总共有N个数据集，将gradient_w和gradient_b求和后除以N，获得数据集的平均gradient_w和gradient_b
    gradient_w = 0
    gradient_b = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        gradient_w+= (2/N)*((w_current*x+b_current)-y)*x
        gradient_b+= (2/N)*((w_current*x+b_current)-y)
#     用当前的b减去b的梯度，用当前w减去w的梯度
    new_w = w_current - learningRate*gradient_w
    new_b = b_current - learningRate*gradient_b
    return new_b,new_w


# 批量更新迭代w和b的值
def gradient_descent_runner(points,starting_b,starting_w,learning_rate,num_iterations):
    b = starting_b
    w = starting_w
    for i in tqdm(range(num_iterations)):
        b,w = step_gradient(w,b,points,learning_rate)
    return b,w
def run():
    points = np.genfromtxt('data.csv',delimiter=',')
    learning_rate = 0.0001
    initial_w = 0
    initial_b = 0
    num_iterations = 1000
    print('Starting gradient descent at b = {0},w = {1},error = {2}'.format(initial_b,initial_w,compute_error_line_regression(initial_w,initial_b,points)))
    b,w = gradient_descent_runner(points,initial_b,initial_w,learning_rate,num_iterations)
    print('After{0}iterations b = {1},w = {2},error = {3}'.format(num_iterations,b,w,compute_error_line_regression(w,b,points)))

# 主函数
if __name__ == '__main__':
    # After1000iterations b = 0.08893651993741344,w = 1.4777440851894448,error = 112.61481011613473
    run()





