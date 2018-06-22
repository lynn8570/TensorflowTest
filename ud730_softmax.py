# coding=utf-8
import matplotlib.pyplot as plt

scores = [3.0,1.0,0.2]

import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

print(softmax(scores))


x = np.arange(-2.0,6.0,0.1) #生成一个array，从-0.2到6.0（不包括6,步长为0。1的数组
print(x)

# vstack
scores = np.vstack([x,np.ones_like(x),0.2 * np.ones_like(x)])#三组 x,x形状的1，x形状的0.2.

print("scores",scores)
plt.plot(x,softmax(scores).T,linewidth = 2) #scores有三组，代表三条线。
plt.show();


scores2=np.array([3.0,1.0,0.2])
print(softmax(scores2*10))
print(softmax(scores2/10))
