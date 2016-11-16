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
naveragingtrials = 5
nepochs=1


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

#For active3 network and activeah
def score(x_relevant):
    return (1+numpy.sign(numpy.dot(score_weights,x_relevant)))/2.0

def easiness(x):
    return -(score(x[:nrelevantdim])*2-1)*(numpy.dot(score_weights,x[:nrelevantdim])) #/numpy.linalg.norm(x[:nrelevantdim]) # minus sign so that larger distances come earlier
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
active3_error_track = [0]*naveragingtrials
activenc_error_track = [0]*naveragingtrials
activeah_error_track = [0]*naveragingtrials
ahcurr_error_track = [0]*naveragingtrials

dot_track = [0]*naveragingtrials
standard_dot_track = [0]*naveragingtrials
standard2_dot_track = [0]*naveragingtrials
active_dot_track = [0]*naveragingtrials
active2_dot_track = [0]*naveragingtrials
active3_dot_track = [0]*naveragingtrials
activenc_dot_track = [0]*naveragingtrials
activeah_dot_track = [0]*naveragingtrials
ahcurr_dot_track = [0]*naveragingtrials

#time spent in easier trials vs. initial angle between hyperplanes
active_initial_dot_track = [0]*naveragingtrials
active2_initial_dot_track = [0]*naveragingtrials
active_easy_examples_seen_track = [0]*naveragingtrials
active2_easy_examples_seen_track = [0]*naveragingtrials


#Is there any connection between curriculum and active 1 and 2 and the strategy of active 3?
standard_example_dot_track = numpy.zeros(nepochs*nsamples)
curr_example_dot_track = numpy.zeros(nepochs*nsamples)
active1_example_dot_track = numpy.zeros(nepochs*nsamples)
active2_example_dot_track = numpy.zeros(nepochs*nsamples)
active3_example_dot_track = numpy.zeros(nepochs*nsamples)
activenc_example_dot_track = numpy.zeros(nepochs*nsamples)
activeah_example_dot_track = numpy.zeros(nepochs*nsamples)
ahcurr_example_dot_track = numpy.zeros(nepochs*nsamples)

#How do they progress?
full_dot_track = numpy.zeros(nepochs*nsamples)
curr_full_dot_track = numpy.zeros(nepochs*nsamples)
active1_full_dot_track = numpy.zeros(nepochs*nsamples)
active2_full_dot_track = numpy.zeros(nepochs*nsamples)
active3_full_dot_track = numpy.zeros(nepochs*nsamples)
activenc_full_dot_track = numpy.zeros(nepochs*nsamples)
activeah_full_dot_track = numpy.zeros(nepochs*nsamples)
ahcurr_full_dot_track = numpy.zeros(nepochs*nsamples)

for seed in xrange(naveragingtrials):
    print "On trial: "+str(seed)
    #seed = 1
    tf.set_random_seed(seed)
    numpy.random.seed(seed)
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
    active3_W1 =  tf.Variable(W1.initialized_value())
    activenc_W1 =  tf.Variable(W1.initialized_value())
    activeah_W1 =  tf.Variable(W1.initialized_value())
    ahcurr_W1 =  tf.Variable(W1.initialized_value())
    b1 = tf.Variable(tf.zeros([1,1]))
    standard_b1 = tf.Variable(b1.initialized_value())
    standard2_b1 = tf.Variable(b1.initialized_value())
    active_b1 = tf.Variable(b1.initialized_value())
    active2_b1 = tf.Variable(b1.initialized_value())
    active3_b1 = tf.Variable(b1.initialized_value())
    activenc_b1 = tf.Variable(b1.initialized_value())
    activeah_b1 = tf.Variable(b1.initialized_value())
    ahcurr_b1 = tf.Variable(b1.initialized_value())
    output = tf.nn.sigmoid(tf.matmul(W1,input_ph)+b1)
    standard_output = tf.nn.sigmoid(tf.matmul(standard_W1,input_ph)+standard_b1)
    standard2_output = tf.nn.sigmoid(tf.matmul(standard2_W1,input_ph)+standard2_b1)
    active_output = tf.nn.sigmoid(tf.matmul(active_W1,input_ph)+active_b1)
    active2_output = tf.nn.sigmoid(tf.matmul(active2_W1,input_ph)+active2_b1)
    active3_output = tf.nn.sigmoid(tf.matmul(active3_W1,input_ph)+active3_b1)
    activenc_output = tf.nn.sigmoid(tf.matmul(activenc_W1,input_ph)+activenc_b1)
    activeah_output = tf.nn.sigmoid(tf.matmul(activeah_W1,input_ph)+activeah_b1)
    ahcurr_output = tf.nn.sigmoid(tf.matmul(ahcurr_W1,input_ph)+ahcurr_b1)

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

    active3_output_error = tf.square(active3_output - target_ph)
    active3_output_correct = (1.0-tf.sign((active3_output-0.5)*(target_ph-0.5)))/2.0
    active3_loss = tf.reduce_mean(active3_output_error)
    active3_train = optimizer.minimize(active3_loss)

    activenc_output_error = tf.square(activenc_output - target_ph)
    activenc_output_correct = (1.0-tf.sign((activenc_output-0.5)*(target_ph-0.5)))/2.0
    activenc_loss = tf.reduce_mean(activenc_output_error)
    activenc_train = optimizer.minimize(activenc_loss)

    ahcurr_output_error = tf.square(ahcurr_output - target_ph)
    ahcurr_output_correct = (1.0-tf.sign((ahcurr_output-0.5)*(target_ph-0.5)))/2.0
    ahcurr_loss = tf.reduce_mean(ahcurr_output_error)
    ahcurr_train = optimizer.minimize(ahcurr_loss)

    activeah_output_error = tf.square(activeah_output - target_ph)
    activeah_output_correct = (1.0-tf.sign((activeah_output-0.5)*(target_ph-0.5)))/2.0
    activeah_loss = tf.reduce_mean(activeah_output_error)
    activeah_train = optimizer.minimize(activeah_loss)

    init = tf.initialize_all_variables()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    curr_AH_weights = sess.run(ahcurr_W1)[0]
    x_data_AH_easiness_indices = numpy.argsort(map(lambda x: score(x[:nrelevantdim])*numpy.dot(curr_AH_weights,x),x_data))  
    AH_x_data = x_data[x_data_AH_easiness_indices] #Sort by margin from current boundary
    AH_y_data = numpy.array(map(lambda x: score(x[:nrelevantdim]),AH_x_data))
    AH_y_data = AH_y_data.reshape([nsamples,1])

    def calculate_error_rates():
	error = 0.0
	standard_error = 0.0
	active_error = 0.0
	active2_error = 0.0
	active3_error = 0.0
	activenc_error = 0.0
	ahcurr_error = 0.0
	activeah_error = 0.0
	standard2_error = 0.0
	for sample in xrange(ntestsamples):
	    error += sess.run(output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    standard_error += sess.run(standard_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    standard2_error += sess.run(standard2_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    active_error += sess.run(active_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    activenc_error += sess.run(activenc_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    ahcurr_error += sess.run(ahcurr_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    activeah_error += sess.run(activeah_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    active2_error += sess.run(active2_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	    active3_error += sess.run(active3_output_correct,feed_dict={input_ph: test_x_data[sample].reshape([ndim,1]),target_ph: test_y_data[sample].reshape([1,1])})[0,0]
	error /= ntestsamples
	standard_error /= ntestsamples
	active_error /= ntestsamples
	activenc_error /= ntestsamples
	activeah_error /= ntestsamples
	ahcurr_error /= ntestsamples
	active2_error /= ntestsamples
	active3_error /= ntestsamples
	standard2_error /= ntestsamples
	return error,standard_error,active_error,active2_error,standard2_error,active3_error,activenc_error,ahcurr_error,activeah_error


    #active curriculum stuff
    active_chunks_passed = 0
    active_examples_seen = [0]*nactivechunks
    def show_active_example(chunk):
	if active_examples_seen[chunk] < chunk_size:
	    index = chunk_size*chunk+active_examples_seen[chunk]
	else: #If all examples seen, just pick another at random
	    index = numpy.random.randint(chunk_size*chunk,chunk_size*(chunk+1)) 
	sess.run(active_train,feed_dict={input_ph: sorted_x_data[index].reshape([ndim,1]),target_ph: sorted_y_data[index].reshape([1,1])})
	active_examples_seen[chunk] = active_examples_seen[chunk]+1
	return index

    #active non-curriculum stuff
    activenc_chunks_passed = 0
    activenc_examples_seen = [0]*nactivechunks
    def show_activenc_example(chunk):
	if activenc_examples_seen[chunk] < chunk_size:
	    index = chunk_size*chunk+activenc_examples_seen[chunk]
	else: #If all examples seen, just pick another at random
	    index = numpy.random.randint(chunk_size*chunk,chunk_size*(chunk+1)) 
	sess.run(activenc_train,feed_dict={input_ph: x_data[index].reshape([ndim,1]),target_ph: y_data[index].reshape([1,1])})
	activenc_examples_seen[chunk] = activenc_examples_seen[chunk]+1
	return index

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
	return error,index

    #active3 curriculum stuff
    active3_unseen_example_track = range(nsamples)
    def show_active3_example():
	curr_weights = sess.run(active3_W1)
	#dists = map(lambda i: (1-numpy.dot(score_weights,sorted_x_data[i,:nrelevantdim])/(numpy.linalg.norm(score_weights)*numpy.linalg.norm(sorted_x_data[i,:nrelevantdim])))*numpy.abs(numpy.dot(curr_weights[0],sorted_x_data[i])/(numpy.linalg.norm(curr_weights[0])*numpy.linalg.norm(sorted_x_data[i]))),active3_unseen_example_track)			
	dists = map(lambda i: numpy.abs(numpy.dot(curr_weights[0],sorted_x_data[i])/(numpy.linalg.norm(curr_weights[0])*numpy.linalg.norm(sorted_x_data[i]))),active3_unseen_example_track)			
	index = active3_unseen_example_track.pop(numpy.argmin(dists))
	sess.run(active3_train,feed_dict={input_ph: sorted_x_data[index].reshape([ndim,1]),target_ph: sorted_y_data[index].reshape([1,1])}) 
	return index

    #activeah curriculum stuff
    activeah_last_n = [0]*active2tracklength
    activeah_chunks_passed = 0
    activeah_examples_seen = [0]*nactivechunks
    def show_activeah_example(chunk):
	if activeah_examples_seen[chunk] < chunk_size:
	    index = chunk_size*chunk+activeah_examples_seen[chunk]
	else: #If all examples seen, just pick another at random
	    index = numpy.random.randint(chunk_size*chunk,chunk_size*(chunk+1)) 
	error = sess.run(activeah_output_correct,feed_dict={input_ph: AH_x_data[index].reshape([ndim,1]),target_ph: AH_y_data[index].reshape([1,1])}) 
	sess.run(activeah_train,feed_dict={input_ph: AH_x_data[index].reshape([ndim,1]),target_ph: AH_y_data[index].reshape([1,1])}) #Could be more efficient than running these twice with a placeholder
	activeah_examples_seen[chunk] = activeah_examples_seen[chunk]+1
	return error,index

    def calculate_train_subset_error_rate(): 
	"""Calculates error rates of the active network on this chunk"""
	error = 0.0
	chunk_offset = active_chunks_passed*chunk_size
	for sample in xrange(chunk_offset,chunk_offset+chunk_size):
	    error += sess.run(active_output_correct,feed_dict={input_ph: sorted_x_data[sample].reshape([ndim,1]),target_ph: sorted_y_data[sample].reshape([1,1])})[0,0]
	error = error/chunk_size 
	return error

    these_weights = sess.run(active_W1[0,:nrelevantdim])
    active_initial_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(active2_W1[0,:nrelevantdim])
    active2_initial_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    #training
    for epoch in xrange(nepochs):
	active_chunks_passed = 0
	active_examples_seen = [0]*nactivechunks
	activenc_chunks_passed = 0
	activenc_examples_seen = [0]*nactivechunks
	active2_last_n = [0]*active2tracklength
	active2_chunks_passed = 0
	activeah_last_n = [0]*active2tracklength
	activeah_chunks_passed = 0
	active2_examples_seen = [0]*nactivechunks
	activeah_examples_seen = [0]*nactivechunks
	active3_unseen_example_track = range(nsamples)
	for sample in xrange(nsamples):
	    sess.run(train,feed_dict={input_ph: sorted_x_data[sample].reshape([ndim,1]),target_ph: sorted_y_data[sample].reshape([1,1])})
	    curr_weights = sess.run(W1)
	    curr_example_dot_track[sample] += numpy.abs(numpy.dot(curr_weights[0],sorted_x_data[sample])/(numpy.linalg.norm(curr_weights[0])*numpy.linalg.norm(sorted_x_data[sample])))
	    sess.run(standard_train,feed_dict={input_ph: x_data[sample].reshape([ndim,1]),target_ph: y_data[sample].reshape([1,1])})
	    curr_weights = sess.run(standard_W1)
	    standard_example_dot_track[sample] += numpy.abs(numpy.dot(curr_weights[0],x_data[sample])/(numpy.linalg.norm(curr_weights[0])*numpy.linalg.norm(x_data[sample])))
	    sess.run(standard2_train,feed_dict={input_ph: standard2_x_data[sample].reshape([ndim,1]),target_ph: standard2_y_data[sample].reshape([1,1])})
	    #Active training
	    if (active_chunks_passed < nactivechunks-1) and (sample % active_sample_test_period == 0): 
		active_error = calculate_train_subset_error_rate()
		if active_error < active_error_threshold or (active_examples_seen[active_chunks_passed] > 0.5*nsamples):
		    active_chunks_passed += 1
	    a_index = show_active_example(active_chunks_passed)
	    curr_weights = sess.run(active_W1)
	    active1_example_dot_track[sample+epoch*nsamples] += numpy.abs(numpy.dot(curr_weights[0],sorted_x_data[a_index])/(numpy.linalg.norm(curr_weights[0])*numpy.linalg.norm(sorted_x_data[a_index])))
	    #Activenc training
	    if (activenc_chunks_passed < nactivechunks-1) and (sample % active_sample_test_period == 0): 
		activenc_error = calculate_train_subset_error_rate()
		if activenc_error < active_error_threshold or (activenc_examples_seen[activenc_chunks_passed] > 0.5*nsamples):
		    activenc_chunks_passed += 1
	    a_index = show_activenc_example(activenc_chunks_passed)
	    curr_weights = sess.run(activenc_W1)
	    activenc_example_dot_track[sample+epoch*nsamples] += numpy.abs(numpy.dot(curr_weights[0],sorted_x_data[a_index])/(numpy.linalg.norm(curr_weights[0])*numpy.linalg.norm(sorted_x_data[a_index])))
	    #Active2 training
	    if  (active2_chunks_passed < nactivechunks-1) and (sum(active2_last_n) >= active2numtopass or  (active2_examples_seen[active2_chunks_passed] > 0.5*nsamples)): 
		active2_chunks_passed += 1
		active2_last_n = [0]*active2tracklength
	    this_a2_error,a_index = show_active2_example(active2_chunks_passed)
	    active2_last_n.pop(0)
	    active2_last_n.append(1-this_a2_error)
	    curr_weights = sess.run(active2_W1)
	    active2_example_dot_track[sample+epoch*nsamples] += numpy.abs(numpy.dot(curr_weights[0],sorted_x_data[a_index])/(numpy.linalg.norm(curr_weights[0])*numpy.linalg.norm(sorted_x_data[a_index])))
	    #Active3 training
	    a_index = show_active3_example()
	    curr_weights = sess.run(active3_W1)
	    active3_example_dot_track[sample+epoch*nsamples] += numpy.abs(numpy.dot(curr_weights[0],sorted_x_data[a_index])/(numpy.linalg.norm(curr_weights[0])*numpy.linalg.norm(sorted_x_data[a_index])))
	    #AHcurr training
	    curr_weights = sess.run(ahcurr_W1)[0]
	    sess.run(ahcurr_train,feed_dict={input_ph: AH_x_data[sample].reshape([ndim,1]),target_ph: AH_y_data[sample].reshape([1,1])})
	    curr_weights = sess.run(ahcurr_W1)
	    ahcurr_example_dot_track[sample] += numpy.abs(numpy.dot(curr_weights,AH_x_data[sample])/(numpy.linalg.norm(curr_weights)*numpy.linalg.norm(AH_x_data[sample])))
	    #active AH training
	    if (activeah_chunks_passed < nactivechunks-1) and (sum(activeah_last_n) >= active2numtopass or  (activeah_examples_seen[activeah_chunks_passed] > 0.5*nsamples)): 
		activeah_chunks_passed += 1
		activeah_last_n = [0]*active2tracklength
	    this_aa_error,a_index = show_activeah_example(activeah_chunks_passed)
	    activeah_last_n.pop(0)
	    activeah_last_n.append(1-this_aa_error)
	    curr_weights = sess.run(activeah_W1)
	    activeah_example_dot_track[sample+epoch*nsamples] += numpy.abs(numpy.dot(curr_weights[0],AH_x_data[a_index])/(numpy.linalg.norm(curr_weights[0])*numpy.linalg.norm(AH_x_data[a_index])))
	    #Plot progression of weights
	    these_weights = sess.run(standard_W1[0,:nrelevantdim])
	    full_dot_track[sample+epoch*nsamples] +=  numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
	    these_weights = sess.run(W1[0,:nrelevantdim])
	    curr_full_dot_track[sample+epoch*nsamples] += numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
	    these_weights = sess.run(active_W1[0,:nrelevantdim])
	    active1_full_dot_track[sample+epoch*nsamples] +=  numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
	    these_weights = sess.run(activenc_W1[0,:nrelevantdim])
	    activenc_full_dot_track[sample+epoch*nsamples] +=  numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
	    these_weights = sess.run(activeah_W1[0,:nrelevantdim])
	    activeah_full_dot_track[sample+epoch*nsamples] +=  numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
	    these_weights = sess.run(ahcurr_W1[0,:nrelevantdim])
	    ahcurr_full_dot_track[sample+epoch*nsamples] +=  numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
	    these_weights = sess.run(active2_W1[0,:nrelevantdim])
	    active2_full_dot_track[sample+epoch*nsamples] +=  numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
	    these_weights = sess.run(active3_W1[0,:nrelevantdim])
	    active3_full_dot_track[sample+epoch*nsamples] +=  numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
	    

    errors = calculate_error_rates()
    standard_error_track[seed] = errors[1]
    error_track[seed] = errors[0]
    active_error_track[seed] = errors[2]
    active2_error_track[seed] = errors[3]
    standard2_error_track[seed] = errors[4]
    active3_error_track[seed] = errors[5]
    activenc_error_track[seed] = errors[6]
    ahcurr_error_track[seed] = errors[7]
    activeah_error_track[seed] = errors[8]

    #Weight angles
    these_weights = sess.run(standard_W1[0,:nrelevantdim])
    standard_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(W1[0,:nrelevantdim])
    dot_track[seed] =  numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(active_W1[0,:nrelevantdim])
    active_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(activenc_W1[0,:nrelevantdim])
    activenc_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(ahcurr_W1[0,:nrelevantdim])
    ahcurr_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(active2_W1[0,:nrelevantdim])
    active2_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(standard2_W1[0,:nrelevantdim])
    standard2_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(active3_W1[0,:nrelevantdim])
    active3_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)
    these_weights = sess.run(activeah_W1[0,:nrelevantdim])
    activeah_dot_track[seed] = numpy.dot(these_weights/numpy.linalg.norm(these_weights),normed_weights)

    print("Active example counts: ", active_examples_seen)
    print("Active2 example counts: ", active2_examples_seen)
    print("ActiveAH example counts: ", activeah_examples_seen)
    active_easy_examples_seen_track[seed] = active_examples_seen[0]+active_examples_seen[1]
    active2_easy_examples_seen_track[seed] = active2_examples_seen[0]+active2_examples_seen[1]
    sess.close()
    tf.reset_default_graph()

print("Mean error rates (std. dev.): active curriculum = %f (%f), active2 curriculum = %f (%f), curriculum = %f (%f), non-curriculum = %f (%f), non-curriculum no hard examples = %f (%f), active non-curriculum = %f (%f), ad-hoc curr %f (%f), ad-hoc active %f (%f)" %(numpy.mean(active_error_track),numpy.std(active_error_track),numpy.mean(active2_error_track),numpy.std(active2_error_track),numpy.mean(error_track),numpy.std(error_track),numpy.mean(standard_error_track),numpy.std(standard_error_track),numpy.mean(standard2_error_track),numpy.std(standard2_error_track),numpy.mean(activenc_error_track),numpy.std(activenc_error_track),numpy.mean(ahcurr_error_track),numpy.std(ahcurr_error_track),numpy.mean(activeah_error_track),numpy.std(activeah_error_track)))

numpy.savetxt('s_final_error_AH_on_previous.csv',full_error_track,delimiter=',')
numpy.savetxt('c_final_error_AH_on_previous.csv',curr_full_error_track,delimiter=',')
numpy.savetxt('ah_final_error_AH_on_previous.csv',ahcurr_full_error_track,delimiter=',')

plot.hist([standard_error_track,error_track,active_error_track,active2_error_track,standard2_error_track,ahcurr_error_track,activeah_error_track],histtype='bar')
plot.title("Error rates")
plot.legend(['Non Curr.','Curr.','Act. Curr.','Act2. Curr.','NCNH','AH Curr.','AH Act.'])
plot.show()

plot.hist([standard_dot_track,dot_track,active_dot_track,active2_dot_track,standard2_dot_track,ahcurr_dot_track,activeah_dot_track],histtype='bar')
plot.title("Weight vec. dot values")
plot.legend(['Non Curr.','Curr.','Act. Curr.','Act2. Curr.','NCNH','AH Curr.','AH Act.'])
plot.show()

#testing the relationship between initial angle and easy examples seen
print numpy.corrcoef(active_initial_dot_track,active_easy_examples_seen_track)
print numpy.corrcoef(active2_initial_dot_track,active2_easy_examples_seen_track)

fig = plot.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(active_initial_dot_track,active_easy_examples_seen_track,c='b',label='active')
ax1.scatter(active2_initial_dot_track,active2_easy_examples_seen_track,c='r',label='active2')
plot.xlabel('Initial cosine of angle')
plot.ylabel('# easy samples seen')

plot.legend()
plot.show()

#Progression of the weights
full_dot_track /= naveragingtrials
curr_full_dot_track /= naveragingtrials
active1_full_dot_track /= naveragingtrials
active2_full_dot_track /= naveragingtrials
active3_full_dot_track /= naveragingtrials
#activenc_full_dot_track /= naveragingtrials
ahcurr_full_dot_track /= naveragingtrials
activeah_full_dot_track /= naveragingtrials

numpy.savetxt('s_dot_track_AH_on_previous.csv',full_dot_track,delimiter=',')
numpy.savetxt('c_dot_track_AH_on_previous.csv',curr_full_dot_track,delimiter=',')
numpy.savetxt('ah_dot_track_AH_on_previous.csv',ahcurr_full_dot_track,delimiter=',')


plot.plot(range(nsamples),full_dot_track,label='Standard')
plot.plot(range(nsamples),curr_full_dot_track,label='Curriculum')
plot.plot(range(nsamples),active1_full_dot_track,label='Act. 1')
plot.plot(range(nsamples),active2_full_dot_track,label='Act. 2')
plot.plot(range(nsamples),active3_full_dot_track,label='Act. 3')
#plot.plot(range(nsamples),activenc_full_dot_track,label='Act. NC')
plot.plot(range(nsamples),ahcurr_full_dot_track,label='AH Curr.')
plot.plot(range(nsamples),activeah_full_dot_track,label='AH Act.')
plot.legend()
plot.xlabel('examples seen')
plot.ylabel('Weight vec. dot value')
plot.show()


standard_example_dot_track /= naveragingtrials
curr_example_dot_track /= naveragingtrials
active1_example_dot_track /= naveragingtrials
active2_example_dot_track /= naveragingtrials
active3_example_dot_track /= naveragingtrials  
#activenc_example_dot_track /= naveragingtrials  
ahcurr_example_dot_track /= naveragingtrials  
activeah_example_dot_track /= naveragingtrials  
plot.plot(range(200),standard_example_dot_track,label='Non-Curr.')
plot.plot(range(200),curr_example_dot_track,label='Curr.')
plot.plot(range(200),active1_example_dot_track,label='active1')
plot.plot(range(200),active2_example_dot_track,label='active2')
plot.plot(range(200),active3_example_dot_track,label='active3')
#plot.plot(range(200),activenc_example_dot_track,label='activenc')
plot.plot(range(200),ahcurr_example_dot_track,label='ahcurr')
plot.plot(range(200),activeah_example_dot_track,label='activeah')
plot.legend()
plot.xlabel('training example')
plot.ylabel('example dot current weights')
plot.show()

#standard_error_track = numpy.array(standard_error_track)
#standard2_error_track = numpy.array(standard2_error_track)
#error_track = numpy.array(error_track)
#active_error_track = numpy.array(active_error_track)
#active2_error_track = numpy.array(active2_error_track)
#
#standard_dot_track = numpy.array(standard_dot_track)
#standard2_dot_track = numpy.array(standard2_dot_track)
#dot_track = numpy.array(dot_track)
#active_dot_track = numpy.array(active_dot_track)
#active2_dot_track = numpy.array(active2_dot_track)
#
#plot.hist([standard_error_track-error_track,error_track-active_error_track,standard_error_track-active2_error_track,error_track-active2_error_track,standard2_error_track-active2_error_track,standard_error_track-standard2_error_track],histtype='bar')
#plot.title("Error rate differences")
#plot.legend(['Non. Curr. - Curr.','Curr - Act. Curr','Non Curr. - Act.2 Curr.','Curr - Act2. Curr.','NCNH - Act2','Non. Curr-NCNH'])
#plot.show()
#
#
#plot.hist([standard_dot_track-dot_track,dot_track-active_dot_track,standard_dot_track-active2_dot_track,dot_track-active2_dot_track,standard2_dot_track - active2_dot_track,standard_dot_track - standard2_dot_track],histtype='bar')
#plot.title("Weight vec. dot differences")
#plot.legend(['Non. Curr. - Curr.','Curr - Act. Curr','Non Curr. - Act.2 Curr.','Curr - Act2. Curr.','NCNH - Act2','Non. Curr. - NCNH'])
#plot.show()
#

