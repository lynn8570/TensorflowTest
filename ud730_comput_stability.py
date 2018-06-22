# coding=utf-8
# 计算稳定性，当一个很大的值，与一个很小的值进行操作的时候，比较容易出现问题。
# 测试：一个10亿+ 10的-6次方的数，加10的6次方 次再减去10亿
# 按理说应该是1 ，但是我们得到的是一个0.953674316406

var=1000000000
for i in xrange(1000000):
    var = var + 1e-6
print(var - 1000000000)


# 所以，再计算损失函数的时候，不希望数值太大或者太小
# 所以，我们需要对输入进行正则化，
# 对输入值进行优化

#最好就是 变量的平均值为 0 ，切变量具有同方差
#具体到图片上，可以理解为 rgb 每个通道的值为 0-255
# 那么 rgb 分别为 (R-128)/128. 这样，内容没变，但是输入值的分布优化了


#另外后一个问题就是 w,b 的初始值
