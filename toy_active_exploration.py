import numpy
import tensorflow as tf
import matplotlib.pyplot as plot


normalize = lambda x: numpy.array(x)/numpy.linalg.norm(x)

true_weights = numpy.array([1/numpy.sqrt(2),1/numpy.sqrt(2)])
easy_example = numpy.array([[10],[10]])
easy_example2 = numpy.array([[-10],[-5]])

hard_example = numpy.array([[10],[-9]])
hard_example2 = numpy.array([[-5],[4.3]])
initial_weights = numpy.array([[1,-2],[2,-4],[3,-6],[4,-8],[5,-10]])


trial_tracks = []
trial_magnitudes = []
trial_final_errors = []
trial_type = [i % 2 for i in xrange(10000)]

#fig = plot.figure()
#ax = fig.add_subplot(111)
for trial in xrange(10000): #(2*len(initial_weights)):
    print(trial)
    tf.set_random_seed(trial//2)
    numpy.random.seed(trial//2)
    input_ph = tf.placeholder(tf.float32, shape=[2,1])
    target_ph = tf.placeholder(tf.float32, shape=[1,1])
    W = tf.Variable(tf.random_normal([1,2],0,1))
#    assign_weights = W.assign(initial_weights[trial//2].reshape([1,2]))
    trial_track = []
    output = tf.nn.sigmoid(tf.matmul(W,input_ph))
    
    output_error = tf.square(output - target_ph)
    output_correct = (1.0-tf.sign((output-0.5)*(target_ph-0.5)))/2.0
    loss = tf.reduce_mean(output_error)
    optimizer = tf.train.GradientDescentOptimizer(1.0)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()
    

    sess = tf.Session()
    sess.run(init)
#    sess.run(assign_weights)


    


    trial_track.append((sess.run(W)))

    if trial % 2 == 0:
	trial_magnitudes.append(numpy.linalg.norm(trial_track[-1]))

	sess.run(train,feed_dict={input_ph: easy_example,target_ph: numpy.array([[1]])})
	trial_track.append((sess.run(W)))
	sess.run(train,feed_dict={input_ph: easy_example2,target_ph: numpy.array([[-1]])})
	trial_track.append((sess.run(W)))
	sess.run(train,feed_dict={input_ph: hard_example,target_ph: numpy.array([[1]])})
	trial_track.append((sess.run(W)))
	sess.run(train,feed_dict={input_ph: hard_example2,target_ph: numpy.array([[-1]])})
    else:
	sess.run(train,feed_dict={input_ph: hard_example,target_ph: numpy.array([[1]])})
	trial_track.append((sess.run(W)))
	sess.run(train,feed_dict={input_ph: hard_example2,target_ph: numpy.array([[-1]])})
	trial_track.append((sess.run(W)))
	trial_magnitudes.append(numpy.linalg.norm(trial_track[-1]))
	sess.run(train,feed_dict={input_ph: easy_example,target_ph: numpy.array([[1]])})
	trial_track.append((sess.run(W)))
	sess.run(train,feed_dict={input_ph: easy_example2,target_ph: numpy.array([[-1]])})

    trial_track.append((sess.run(W)))
    trial_final_errors.append(1-numpy.dot(normalize(trial_track[-1]),true_weights)[0])
    sess.close()
    tf.reset_default_graph()
    trial_track = numpy.array(trial_track)
#    if trial % 2 == 0:
#	ax.plot(trial_track[:,0,0],trial_track[:,0,1],'g-')
#	ax.plot(trial_track[4,0,0],trial_track[4,0,1],'g*')
#	ax.plot(trial_track[2,0,0],trial_track[2,0,1],'go')
#    else:
#	ax.plot(trial_track[:,0,0],trial_track[:,0,1],'r-')
#	ax.plot(trial_track[4,0,0],trial_track[4,0,1],'r*')
#	ax.plot(trial_track[2,0,0],trial_track[2,0,1],'ro')
#
#ax.plot(true_weights[0],true_weights[1],'b*')
#plot.show()


numpy.savetxt('trial_final_errors.csv',trial_final_errors)
numpy.savetxt('trial_type.csv',trial_type)
numpy.savetxt('trial_magnitudes.csv',trial_magnitudes)

print numpy.corrcoef(trial_final_errors,trial_type)
plot.scatter(trial_magnitudes,trial_final_errors,c=trial_type)
plot.legend()
plot.show()
