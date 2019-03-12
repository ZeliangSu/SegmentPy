import tensorflow as tf
import numpy as np

################################################ Neurons layers#########################################################
# input_size = 784
# hidden_layer_neurons = 10
# output_size = 784
# learning_rate = 0.001
# epoch = 1000
# batch_size = 5
#
# X = tf.placeholder(tf.float32, [None, input_size], name="input_X")
# y = tf.placeholder(tf.float32, [None, output_size], name="Output_y")
# X_img = tf.reshape(X, [-1, 28, 28, 1])
# y_img = tf.reshape(X, [-1, 28, 28, 1])
# tf.summary.image('X_img', X_img, 1)
# tf.summary.image('y_img', y_img, 1)
#
# # First layer of weights
# with tf.name_scope("layer1"):
#     W1 = tf.get_variable("W1", shape=[input_size, hidden_layer_neurons],
#                          initializer=tf.contrib.layers.xavier_initializer())
#     layer1 = tf.matmul(X, W1)
#     layer1_act = tf.nn.tanh(layer1)
#     tf.summary.histogram("weights", W1)
#     tf.summary.histogram("layer", layer1)
#     tf.summary.histogram("activations", layer1_act)
#
# # Second layer of weights
# with tf.name_scope("layer2"):
#     W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, hidden_layer_neurons],
#                          initializer=tf.contrib.layers.xavier_initializer())
#     layer2 = tf.matmul(layer1_act, W2)
#     layer2_act = tf.nn.tanh(layer2)
#     tf.summary.histogram("weights", W2)
#     tf.summary.histogram("layer", layer2)
#     tf.summary.histogram("activations", layer2_act)
#
# # Third layer of weights
# with tf.name_scope("layer3"):
#     W3 = tf.get_variable("W3", shape=[hidden_layer_neurons, hidden_layer_neurons],
#                          initializer=tf.contrib.layers.xavier_initializer())
#     layer3 = tf.matmul(layer2_act, W3)
#     layer3_act = tf.nn.tanh(layer3)
#
#     tf.summary.histogram("weights", W3)
#     tf.summary.histogram("layer", layer3)
#     tf.summary.histogram("activations", layer3_act)
#
# # Fourth layer of weights
# with tf.name_scope("layer4"):
#     W4 = tf.get_variable("W4", shape=[hidden_layer_neurons, output_size],
#                          initializer=tf.contrib.layers.xavier_initializer())
#     Qpred = tf.nn.softmax(tf.matmul(layer3_act, W4)) # Bug fixed: Qpred = tf.nn.softmax(tf.matmul(layer3, W4))
#     tf.summary.histogram("weights", W4)
#     tf.summary.histogram("Qpred", Qpred)# First layer of weights


##############################################Convolution layer ########################################################
input_size = 784
hidden_layer_neurons = 10
output_size = 784
learning_rate = 0.001
epoch = 1000
batch_size = 5

X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input_X")
y = tf.placeholder(tf.float32, [None, 14, 14, 1], name="Output_y")
X_img = tf.reshape(X, [-1, 28, 28, 1])
y_img = tf.reshape(X, [-1, 28, 28, 1])
tf.summary.image('X_img', X_img, 1)
tf.summary.image('y_img', y_img, 1)

# C1
with tf.name_scope("layer1"):
    W1 = tf.get_variable("W1", shape=[3, 3, 1, 32],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
    layer1_act = tf.nn.relu(layer1)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("layer", layer1)
    tf.summary.histogram("activations", layer1_act)

# C2
with tf.name_scope("layer2"):
    W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
    layer2 = tf.nn.conv2d(layer1_act, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
    layer2_act = tf.nn.relu(layer2)
    tf.summary.histogram("weights", W2)
    tf.summary.histogram("layer", layer2)
    tf.summary.histogram("activations", layer2_act)

# max pool
with tf.name_scope("maxpool"):
    maxpool = tf.nn.max_pool(layer2_act, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

# C3
with tf.name_scope("layer3"):
    W3 = tf.get_variable("W3", shape=[3, 3, 64, 32],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
    layer3 = tf.nn.conv2d(maxpool, W3, strides=[1, 1, 1, 1], padding='SAME') + b3
    layer3_act = tf.nn.relu(layer3)

    tf.summary.histogram("weights", W3)
    tf.summary.histogram("layer", layer3)
    tf.summary.histogram("activations", layer3_act)

# C4
with tf.name_scope("layer4"):
    W4 = tf.get_variable("W4", shape=[3, 3, 32, 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
    Qpred = tf.nn.conv2d(layer3_act, W4, strides=[1, 1, 1, 1], padding='SAME') + b4
    tf.summary.histogram("weights", W4)
    tf.summary.histogram("Qpred", Qpred)

# Loss function
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.losses.mean_squared_error(
        labels=tf.cast(y, tf.int32),
        predictions=Qpred))
    tf.summary.scalar("Q", tf.reduce_mean(Qpred))
    tf.summary.scalar("Y", tf.reduce_mean(y))
    tf.summary.scalar("loss", loss)

# Learning
# with tf.name_scope("performance"):
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))
grads = optimizer.compute_gradients(loss)
summ_grad = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
train_op = optimizer.minimize(loss)


merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./dum_logs/', sess.graph, 3)
    for i in range(epoch // batch_size):
        print(i)
        X_batch = np.random.rand(784 * 5).reshape(5, 28, 28, 1)
        y_batch = np.random.rand(784 // 4 * 5).reshape(5, 14, 14, 1)
        sum, _, grad_vals = sess.run([merged, train_op, summ_grad], feed_dict={X: X_batch, y: y_batch})
        writer.add_summary(sum, i)

