# coding=utf-8
import tensorflow as tf
import numpy as np

#在0-1的范围内产生一个 [100个0-1分布],[100个0-1分布] 形状的数组。
x_data=np.float32(np.random.rand(2,100))

#np.dot 矩阵乘法,[0.100,0.200]点积 [100个0-1],[100个0-1],最后y_data是一个大小为100的数组列表[]
y_data=np.dot([0.100,0.200],x_data)+0.300


#构造一个线性模型

b = tf.Variable(tf.zeros([1]))#元素初始为0的张量，形状为[1]
W = tf.Variable(tf.random_uniform([1,2],-0.1,1.0))#形状为[1,2]的最小值-0.1最大值为1 的随机分布
y = tf.matmul(W,x_data)+b #matrix multiple y 是一个大小为100的 add b=0 的tensor


#最小方差
loss = tf.reduce_mean(tf.square(y-y_data)) #y - y_data 的方差
optimizer = tf.train.GradientDescentOptimizer(0.5)  #梯度
train = optimizer.minimize(loss) #方差最小化


# 初始化变量
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


for step in xrange(0,201): #0，200 数列
    sess.run(train)
    if step % 20 == 0:
        print step,sess.run(W), sess.run(b)


#最后得到最佳的拟合效果







