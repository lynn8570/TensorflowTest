# coding=utf-8
import tensorflow as tf

my_images = tf.zeros([10,299,299,3])
my_iamge_rank = tf.rank(my_images)
print(my_images)
print(my_iamge_rank)

squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
print(squarish_squares)
print(rank_of_squares)





rank_three_tensor = tf.ones([3, 4, 5]) #三阶 元素为1
matrix = tf.reshape(rank_three_tensor, [6, 10]) #保持元素不变，变形状
element = matrix[2,3]
print("element :" ,element)
zeros = tf.zeros(matrix.shape[1])#形状相同的？
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor
# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.

# yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!

# squarish_squares.shape(1) 出错


int_tensor = tf.constant([1,2,3])
float_tensor = tf.cast( int_tensor,dtype=tf.float32)
print(int_tensor)
print(float_tensor)


t1 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
print(tf.rank(t1))  # 3

t1 = tf.Print(t1,[t1])
result = t1+1




a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)

