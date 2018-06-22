import tensorflow as tf

output = None

hidden_layer_weights =[
    [0.1,0.2,0.4],
    [0.4,0.6,0.6],
    [0.5,0.9,0.1],
    [0.8,0.2,0.8]]
out_weight = [
    [0.1,0.6],
    [0.2,0.1],
    [0.7,0.9]]

weight = [tf.Variable(hidden_layer_weights),
          tf.Variable(out_weight)]
biases = [tf.Variable(tf.zeros(3)),
          tf.Variable(tf.zeros(2))]


features = tf.Variable([[1.0,2.0,3.0,4.0],
                       [-1.0,-2.0,-3.0,-4.0],
                       [11.0,12.0,13.0,14.0]])

# create model
hidden_layer = tf.add(tf.matmul(features,weight[0]),biases[0])

hidden_layer = tf.nn.relu(hidden_layer)

logits = tf.add(tf.matmul(hidden_layer,weight[1]),biases[1])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(logits))
