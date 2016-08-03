import numpy
import tensorflow as tf
import matplotlib.pyplot as plot
from tensorflow.examples.tutorials.mnist import input_data

#Code edited from tensorflow's tutorial

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#####Parameters
size_clean_chunk = 2000
size_light_chunk = 4000
size_med_chunk = 4000
light_theta = 0.1
med_theta = 0.2
heavy_theta = 0.3

batch_size = 50

#####
begin_light_chunk = size_clean_chunk
begin_med_chunk = begin_light_chunk+size_light_chunk
begin_heavy_chunk = begin_med_chunk+size_med_chunk

numpy.random.seed(1)
tf.set_random_seed(1)

standard_order = numpy.random.permutation(len(mnist.train.images)) #first size_clean_chunk images will be clean, etc. 


#noise for mnist
def noise(M,theta): 
    """M a matrix, theta a parameter controlling noise (heavy noise but still some possibility of discrimination is around theta = 0.33), uses both binary noise and gaussian."""
    s = numpy.shape(M)
    return M+numpy.random.binomial(1,theta,s)+theta*numpy.random.standard_normal(s)

#add noise to training data according to chunks
mnist.train.images[begin_light_chunk:begin_med_chunk] = noise(mnist.train.images[begin_light_chunk:begin_med_chunk],light_theta)
mnist.train.images[begin_med_chunk:begin_heavy_chunk] = noise(mnist.train.images[begin_med_chunk:begin_heavy_chunk],med_theta)
mnist.train.images[begin_heavy_chunk:] = noise(mnist.train.images[begin_heavy_chunk:],heavy_theta)
#add noise to testing data
mnist.test.images[:] = noise(mnist.test.images[:],heavy_theta)



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
keep_prob = tf.placeholder(tf.float32)

#standard
s_W_conv1 = weight_variable([5, 5, 1, 32])
s_b_conv1 = bias_variable([32])
s_W_conv2 = weight_variable([5, 5, 32, 64])
s_b_conv2 = bias_variable([64])
s_W_fc1 = weight_variable([7 * 7 * 64, 1024])
s_b_fc1 = bias_variable([1024])
s_W_fc2 = weight_variable([1024, 10])
s_b_fc2 = bias_variable([10])

s_h_conv1 = tf.nn.relu(conv2d(x_image, s_W_conv1) + s_b_conv1)
s_h_pool1 = max_pool_2x2(s_h_conv1)
s_h_conv2 = tf.nn.relu(conv2d(s_h_pool1, s_W_conv2) + s_b_conv2)
s_h_pool2 = max_pool_2x2(s_h_conv2)
s_h_pool2_flat = tf.reshape(s_h_pool2, [-1, 7*7*64])
s_h_fc1 = tf.nn.relu(tf.matmul(s_h_pool2_flat, s_W_fc1) + s_b_fc1)
s_h_fc1_drop = tf.nn.dropout(s_h_fc1, keep_prob)
s_y_conv=tf.nn.softmax(tf.matmul(s_h_fc1_drop, s_W_fc2) + s_b_fc2)

s_cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(s_y_conv), reduction_indices=[1]))
s_train_step = tf.train.AdamOptimizer(1e-4).minimize(s_cross_entropy)
s_correct_prediction = tf.equal(tf.argmax(s_y_conv,1), tf.argmax(y_,1))
s_accuracy = tf.reduce_mean(tf.cast(s_correct_prediction, tf.float32))


#curriculum
c_W_conv1 = tf.Variable(s_W_conv1.initialized_value()) 
c_b_conv1 = tf.Variable(s_b_conv1.initialized_value())  
c_W_conv2 = tf.Variable(s_W_conv2.initialized_value())  
c_b_conv2 = tf.Variable(s_b_conv2.initialized_value())  
c_W_fc1 = tf.Variable(s_W_fc1.initialized_value())
c_b_fc1 = tf.Variable(s_b_fc1.initialized_value())
c_W_fc2 = tf.Variable(s_W_fc2.initialized_value())
c_b_fc2 = tf.Variable(s_b_fc2.initialized_value())

c_h_conv1 = tf.nn.relu(conv2d(x_image, c_W_conv1) + c_b_conv1)
c_h_pool1 = max_pool_2x2(c_h_conv1)
c_h_conv2 = tf.nn.relu(conv2d(c_h_pool1, c_W_conv2) + c_b_conv2)
c_h_pool2 = max_pool_2x2(c_h_conv2)
c_h_pool2_flat = tf.reshape(c_h_pool2, [-1, 7*7*64])
c_h_fc1 = tf.nn.relu(tf.matmul(c_h_pool2_flat, c_W_fc1) + c_b_fc1)
c_h_fc1_drop = tf.nn.dropout(c_h_fc1, keep_prob)
c_y_conv=tf.nn.softmax(tf.matmul(c_h_fc1_drop, c_W_fc2) + c_b_fc2)

c_cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(c_y_conv), reduction_indices=[1]))
c_train_step = tf.train.AdamOptimizer(1e-4).minimize(c_cross_entropy)
c_correct_prediction = tf.equal(tf.argmax(c_y_conv,1), tf.argmax(y_,1))
c_accuracy = tf.reduce_mean(tf.cast(c_correct_prediction, tf.float32))



sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(len(mnist.train.images)//batch_size):
  batch_x = mnist.train.images[standard_order[batch_size*i:batch_size*(i+1)]]
  batch_y = mnist.train.labels[standard_order[batch_size*i:batch_size*(i+1)]]
  if i%100 == 0:
    train_accuracy = s_accuracy.eval(feed_dict={
        x:batch_x, y_: batch_y, keep_prob: 1.0})
    print("step %d, standard training accuracy %g"%(i, train_accuracy))
    train_accuracy = c_accuracy.eval(feed_dict={
        x:batch_x, y_: batch_y, keep_prob: 1.0})
    print("step %d, curriculum training accuracy %g"%(i, train_accuracy))
  s_train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
  #curriculum
  batch_x = mnist.train.images[batch_size*i:batch_size*(i+1)]
  batch_y = mnist.train.labels[batch_size*i:batch_size*(i+1)]
  c_train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

print("standard test accuracy %g"%s_accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
print("curriculum test accuracy %g"%c_accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
