import tensorflow as tf
import numpy
import matplotlib.pyplot as plot

#Toy example from Gulcehre & Bengio 
#standard = random training order, standard2 = random training order w/ 50 hardest examples replaced with 50 easiest, curriculum = examples sorted by easiness, active = training in chunks and progress with 95% acc on current chunk or when spent too long, active2 = training in chunks and progress when running count exceeds threshold

#data parameters
ndim = 60
nrelevantdim = 50
nsamples = 200
ntestsamples = 1000
naveragingtrials = 500


#network parameters
eta = 1.0 #learning rate

#for active network
nactivechunks = 4
active_error_threshold = 0.05
chunk_size = nsamples//nactivechunks
active_sample_test_period = 5 #how many samples between tests for moving to next chunk

#For active2 network
active2tracklength = 30
active2numtopass = 28 #Must get this many of the track correct to proceed



def score(x_relevant):
    return (1+numpy.sign(numpy.dot(score_weights,x_relevant)))/2.0

def easiness(x):
    return -(score(x[:nrelevantdim])*2-1)*(numpy.dot(score_weights,x[:nrelevantdim])) # minus sign so that larger distances come earlier
    #return numpy.sum(x[nrelevantdim:] == 0.0)


def random_mask(n):
    mask = numpy.array([0.0]*n+[1.0]*(ndim-(nrelevantdim+n)))
    numpy.random.shuffle(mask)
    return mask

error_track = [0]*naveragingtrials
standard_error_track = [0]*naveragingtrials
standard2_error_track = [0]*naveragingtrials
active_error_track = [0]*naveragingtrials
active2_error_track = [0]*naveragingtrials

dot_track = [0]*naveragingtrials
standard_dot_track = [0]*naveragingtrials
standard2_dot_track = [0]*naveragingtrials
active_dot_track = [0]*naveragingtrials
active2_dot_track = [0]*naveragingtrials

for seed in xrange(naveragingtrials):
    print "On trial: "+str(seed)
    #seed = 1
    tf.set_random_seed(seed+1234) #attempt to be independent from previous sample 
    numpy.random.seed(seed+1234)
    #data
    score_weights = numpy.random.randn(nrelevantdim)
    normed_weights = score_weights/numpy.linalg.norm(score_weights) #For weight angle computations
    x_data = 2*numpy.random.random([nsamples,ndim])-1
    data_mask = numpy.array(map(random_mask,numpy.random.randint(0,(ndim-nrelevantdim)+1,nsamples)))
    x_data[:,nrelevantdim:] = x_data[:,nrelevantdim:]*data_mask

    x_data_easiness_indices = numpy.argsort(map(lambda x: easiness(x),x_data))
    sorted_x_data = x_data[x_data_easiness_indices] #Sort by easiness
    standard2_x_data = numpy.concatenate((sorted_x_data[:150],sorted_x_data[:50]),0) 
    numpy.random.shuffle(standard2_x_data)

    y_data = numpy.array(map(lambda x: score(x[:nrelevantdim]),x_data))
    y_data = y_data.reshape([nsamples,1])

    sorted_y_data = numpy.array(map(lambda x: score(x[:nrelevantdim]),sorted_x_data))
    sorted_y_data = sorted_y_data.reshape([nsamples,1])

    standard2_y_data = numpy.array(map(lambda x: score(x[:nrelevantdim]),standard2_x_data))
    standard2_y_data = standard2_y_data.reshape([nsamples,1])

    test_x_data = 2*numpy.random.random([ntestsamples,ndim])-1
    test_data_mask = numpy.array(map(random_mask,numpy.random.randint(0,(ndim-nrelevantdim)+1,ntestsamples)))
    test_x_data[:,nrelevantdim:] = test_x_data[:,nrelevantdim:]*test_data_mask
    test_y_data = numpy.array(map(lambda x: score(x[:nrelevantdim]),test_x_data))
    test_y_data = test_y_data.reshape([ntestsamples,1])

    #network definitions
    input_ph = tf.placeholder(tf.float32, shape=[ndim,1])
    target_ph =  tf.placeholder(tf.float32, shape=[1,1])
    W1 = tf.Variable(tf.random_normal([1,ndim],0,1))
    standard_W1 = tf.Variable(W1.initialized_value())
    standard2_W1 = tf.Variable(W1.initialized_value())
    active_W1 =  tf.Variable(W1.initialized_value())
    active2_W1 =  tf.Variable(W1.initialized_value())
    b1 = tf.Variable(tf.zeros([1,1]))
    standard_b1 = tf.Variable(b1.initialized_value())
    standard2_b1 = tf.Variable(b1.initialized_value())
    active_b1 = tf.Variable(b1.initialized_value())
    active2_b1 = tf.Variable(b1.initialized_value())
    output = tf.nn.sigmoid(tf.matmul(W1,input_ph)+b1)
    standard_output = tf.nn.sigmoid(tf.matmul(standard_W1,input_ph)+standard_b1)
    standard2_output = tf.nn.sigmoid(tf.matmul(standard2_W1,input_ph)+standard2_b1)
    active_output = tf.nn.sigmoid(tf.matmul(active_W1,input_ph)+active_b1)
    active2_output = tf.nn.sigmoid(tf.matmul(active2_W1,input_ph)+active2_b1)

    output_error = tf.square(output - target_ph)
    output_correct = (1.0-tf.sign((output-0.5)*(target_ph-0.5)))/2.0
    loss = tf.reduce_mean(output_error)
    optimizer = tf.train.GradientDescentOptimizer(eta)
    train = optimizer.minimize(loss)

    standard_output_error = tf.square(standard_output - target_ph)
    standard_output_correct = (1.0-tf.sign((standard_output-0.5)*(target_ph-0.5)))/2.0
    standard_loss = tf.reduce_mean(standard_output_error)
    standard_train = optimizer.minimize(standard_loss)

    standard2_output_error = tf.square(standard2_output - target_ph)
    standard2_output_correct = (1.0-tf.sign((standard2_output-0.5)*(target_ph-0.5)))/2.0
    standard2_loss = tf.reduce_mean(standard2_output_error)
    standard2_train = optimizer.minimize(standard2_loss)

    active_output_error = tf.square(active_output - target_ph)
    active_output_correct = (1.0-tf.sign((active_output-0.5)*(target_ph-0.5)))/2.0
    active_loss = tf.reduce_mean(active_output_error)
    active_train = optimizer.minimize(active_loss)

    active2_output_error = tf.square(active2_output - target_ph)
    active2_output_correct = (1.0-tf.sign((active2_output-0.5)*(target_ph-0.5)))/2.0
    active2_loss = tf.reduce_mean(active2_output_error)
    active2_train = optimizer.minimize(active2_loss)

    init = tf.initialize_all_variables()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    def calculate_error_rates():
	error = 0.0
	standard_error = 0.0
	active_error = 0.0
	active2_error = 0.0
	standard2_error = 0.0
	for sample in xrange(ntestsamples):
	    error += sess.run(output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    standard_error += sess.run(standard_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    standard2_error += sess.run(standard2_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    active_error += sess.run(active_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    active2_error += sess.run(active2_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	error /= ntestsamples
	standard_error /= ntestsamples
	active_error /= ntestsamples
	active2_error /= ntestsamples
	standard2_error /= ntestsamples
	return error,standard_error,active_error,active2_error,standard2_error


    #active curriculum stuff
    active_chunks_passed = 0
    active_examples_seen = [0]*nactivechunks
    def show_active_example(chunk):
	if active_examples_seen[chunk] <chunk_size:
	    index = chunk_size*chunk+active_examples_seen[chunk]
	else: #If all examples seen, just pick another at random
	    index = numpy.random.randint(chunk_size*chunk,chunk_size*(chunk+1)) 
	sess.run(active_train,feed_dict={input_ph: sorted_x_data[index].reshape([ndim,1]),target_ph: sorted_y_data[index].reshape([1,1])})
	active_examples_seen[chunk] = active_examples_seen[chunk]+1


    #active2 curriculum stuff
    active2_last_n = [0]*active2tracklength
    active2_chunks_passed = 0
    active2_examples_seen = [0]*nactivechunks
    def show_active2_example(chunk):
	if active2_examples_seen[chunk] < chunk_size:
	    index = chunk_size*chunk+active2_examples_seen[chunk]
	else: #If all examples seen, just pick another at random
	    index = numpy.random.randint(chunk_size*chunk,chunk_size*(chunk+1)) 
	error = sess.run(active2_output_correct,feed_dict={input_ph: sorted_x_data[index].reshape([ndim,1]),target_ph: sorted_y_data[index].reshape([1,1])}) 
	sess.run(active2_train,feed_dict={input_ph: sorted_x_data[index].reshape([ndim,1]),target_ph: sorted_y_data[index].reshape([1,1])}) #Could be more efficient than running these twice with a placeholder
	active2_examples_seen[chunk] = active2_examples_seen[chunk]+1
	return error

    def calculate_train_subset_error_rate(): 
	"""Calculates error rates of the active network on this chunk"""
	error = 0.0
	chunk_offset = active_chunks_passed*chunk_size
	for sample in xrange(chunk_offset,chunk_offset+chunk_size):
	    error += sess.run(active_output_correct,feed_dict={input_ph: sorted_x_data[sample].reshape([ndim,1]),target_ph: sorted_y_data[sample].reshape([1,1])})[0,0]
	error = error/chunk_size 
	return error

    #training
    for sample in xrange(nsamples):
	sess.run(train,feed_dict={input_ph: sorted_x_data[sample].reshape([ndim,1]),target_ph: sorted_y_data[sample].reshape([1,1])})
	sess.run(standard_train,feed_dict={input_ph: x_data[sample].reshape([ndim,1]),target_ph: y_data[sample].reshape([1,1])})
	sess.run(standard2_train,feed_dict={input_ph: standard2_x_data[sample].reshape([ndim,1]),target_ph: standard2_y_data[sample].reshape([1,1])})
	#Active training
	if (active_chunks_passed < nactivechunks-1) and (sample % active_sample_test_period == 0): 
	    active_error = calculate_train_subset_error_rate()
	    if active_error < active_error_threshold or (active_examples_seen[active_chunks_passed] > 0.5*nsamples):
		active_chunks_passed += 1
	show_active_example(active_chunks_passed)
	#Active2 training
	if  (active2_chunks_passed < nactivechunks-1) and (sum(active2_last_n) >= active2numtopass or  (active2_examples_seen[active2_chunks_passed] > 0.5*nsamples)): 
	    active2_chunks_passed += 1
	    active2_last_n = [0]*active2tracklength
	this_a2_error = show_active2_example(active2_chunks_passed)
	active2_last_n.pop(0)
	active2_last_n.append(1-this_a2_error)
	

    errors = calculate_error_rates()
    standard_error_track[seed] = errors[1]
    error_track[seed] = errors[0]
    active_error_track[seed] = errors[2]
    active2_error_track[seed] = errors[3]
    standard2_error_track[seed] = errors[4]

    #Weight angles
    these_weights = sess.run(standard_W1[0,:nrelevantdim])
    standard_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(W1[0,:nrelevantdim])
    dot_track[seed] =  numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(active_W1[0,:nrelevantdim])
    active_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(active2_W1[0,:nrelevantdim])
    active2_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(standard2_W1[0,:nrelevantdim])
    standard2_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)

    print("Active example counts: ", active_examples_seen)
    print("Active2 example counts: ", active2_examples_seen)
    sess.close()
    tf.reset_default_graph()

print("Mean error rates (std. dev.): active curriculum = %f (%f), active2 curriculum = %f (%f), curriculum = %f (%f), non-curriculum = %f (%f), non-curriculum no hard examples = %f (%f)" %(numpy.mean(active_error_track),numpy.std(active_error_track),numpy.mean(active2_error_track),numpy.std(active2_error_track),numpy.mean(error_track),numpy.std(error_track),numpy.mean(standard_error_track),numpy.std(standard_error_track),numpy.mean(standard2_error_track),numpy.std(standard2_error_track)))

standard_error_track = numpy.array(standard_error_track)
standard2_error_track = numpy.array(standard2_error_track)
error_track = numpy.array(error_track)
active_error_track = numpy.array(active_error_track)
active2_error_track = numpy.array(active2_error_track)

standard_dot_track = numpy.array(standard_dot_track)
standard2_dot_track = numpy.array(standard2_dot_track)
dot_track = numpy.array(dot_track)
active_dot_track = numpy.array(active_dot_track)
active2_dot_track = numpy.array(active2_dot_track)

plot.hist([standard_error_track-error_track,error_track-active_error_track,standard_error_track-active2_error_track,error_track-active2_error_track,standard2_error_track-active2_error_track,standard_error_track-standard2_error_track],histtype='bar')
plot.title("Error rate differences")
plot.legend(['Non. Curr. - Curr.','Curr - Act. Curr','Non Curr. - Act.2 Curr.','Curr - Act2. Curr.','NCNH - Act2','Non. Curr-NCNH'])
plot.show()


plot.hist([standard_dot_track-dot_track,dot_track-active_dot_track,standard_dot_track-active2_dot_track,dot_track-active2_dot_track,standard2_dot_track - active2_dot_track,standard_dot_track - standard2_dot_track],histtype='bar')
plot.title("Weight vec. dot differences")
plot.legend(['Non. Curr. - Curr.','Curr - Act. Curr','Non Curr. - Act.2 Curr.','Curr - Act2. Curr.','NCNH - Act2','Non. Curr. - NCNH'])
plot.show()

