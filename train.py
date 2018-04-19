import tensorflow as tf
import numpy as np
import pandas as pd
import layers as l
from sklearn import metrics


NUMBER_EPOCHS_TO_TRAIN = 1000
BATCH_SIZE = 120

TASK_TRAIN = 'train'
TASK_PREDICT = 'predict'


def model(layers, iterator, process_input_output_fn, kwargs, task=TASK_TRAIN, loss=None):
	"""
	Creates the graph and returns values according to task

	layers: the list of initialized layers
	iterator: iterator over the dataset which gets items in batches
	process_input_output_fn: function to process input and output
	kwargs: dictionary of arguments to be passed to above function
	task: task for which this function is called
	loss: loss to be minimized

	Returns: a tensor of predictions if task is to predict 
			 a list of tensors of grad cost weights and grad cost biases if task is to train
	"""

	next_item = iterator.get_next()

	feature_batch, output_batch = process_input_output_fn(next_item[0], next_item[1], **kwargs)

	pre_calculations = []

	layers[0].features = feature_batch

	pre_calculations.append(layers[0].features)

	if loss is not None:
		loss.outputs = output_batch
		pre_calculations.append(loss.outputs)


	prints = []
	check = []

	for i, cur_layer in enumerate(layers[1:]):
		if cur_layer.has_weights:
			check.append(tf.Print(cur_layer.weights, 
				[cur_layer.weights], message="weights " + str(i+1), 
				summarize=cur_layer.nodes*cur_layer.prev_layer.nodes))

		if cur_layer.has_bias:
			check.append(tf.Print(cur_layer.biases, 
				[cur_layer.biases], message="biases " + str(i+1), summarize=cur_layer.nodes))


	with tf.control_dependencies(pre_calculations):
		# calculate activations of the last layer which will recalculate activations of all the 
		# previous layers
		calc_activations_op = layers[-1].calc_activations()

	with tf.control_dependencies([calc_activations_op]):
		if loss is not None:
			calc_loss = loss.calculate_loss()

		if task == TASK_PREDICT:
			return tf.argmax(calc_activations_op, 1)

	# loss should be there for below calculations

	# set loss to required layers
	for layer in layers:
		if layer.require_loss:
			layer.loss = calc_loss

	with tf.control_dependencies([*check, calc_activations_op, calc_loss]): # was calc_activations_op
		# calculate error terms for the second layers which will calculate error terms for all 
		# the next layers
		calc_grad_cost_pre_activation_op = layers[2].calc_grad_cost_pre_activation_prev_layer()

		# calculate grad activation weight (not useful here)
		calc_grad_activation_weight_op = \
		[layers[i].calc_grad_activation_weight() for i in range(1, len(layers))]


	grad_cost_weights = []
	grad_cost_biases = []

	with tf.control_dependencies([*prints, calc_grad_cost_pre_activation_op, 
		*calc_grad_activation_weight_op]):

		for i, cur_layer in enumerate(layers[1:]):
			if cur_layer.has_weights:
				grad_cost_weights.append(cur_layer.calc_grad_cost_weight())

			if cur_layer.has_bias:
				grad_cost_biases.append(cur_layer.calc_grad_cost_bias())

		return grad_cost_weights, grad_cost_biases


def gradient_descent(process_input_output_fn, kwargs, layers, loss, dataset, train_size, alpha, sess):
	"""
	Perform the gradient descent and updates weights, biases

	process_input_output_fn: function to process input and output
	kwargs: dictionary of arguments to be passed to above function
	layers: the list of initialized layers
	loss: loss to be used
	dataset: the dataset whose iterator returns the input and outputs in batches
	train_size: the size of the dataset that each epoch should at least travel
	alpha: the step_size
	"""

	# run the gradient descent on entire dataset for number of epochs

	n_iter = (train_size / BATCH_SIZE + (train_size % BATCH_SIZE != 0)) * NUMBER_EPOCHS_TO_TRAIN
	print("number of iterations gradient descent will perform:", n_iter)

	iterator = dataset.make_one_shot_iterator()

	cond = lambda i: tf.less(i, n_iter)

	def body(iter_n):
		operations = []
		prints = []

		grad_cost_weights, grad_cost_biases = model(layers, iterator, process_input_output_fn, 
			kwargs, task=TASK_TRAIN, loss=loss)

		calc_loss = loss.get_loss()
		prints.append(tf.Print(calc_loss, [calc_loss], message="loss is "))

		for i, cur_layer in enumerate(layers[1:]):
			if cur_layer.has_weights:
				update = tf.subtract(cur_layer.weights, (alpha / BATCH_SIZE) * (grad_cost_weights[i]))
				operations.append(tf.assign(cur_layer.weights, update))

			if cur_layer.has_bias:
				update = tf.subtract(cur_layer.biases, (alpha / BATCH_SIZE) * (grad_cost_biases[i]))
				operations.append(tf.assign(cur_layer.biases, update))

		with tf.control_dependencies([*prints, *operations]):
			return tf.add(iter_n, 1.0)


	while_loop = tf.while_loop(cond, body, [tf.constant(0.0)], parallel_iterations=1)
	sess.run(while_loop)

	check = []
	for i, cur_layer in enumerate(layers[1:]):
		if cur_layer.has_weights:
			check.append(tf.Print(cur_layer.weights, 
				[cur_layer.weights], message="weights " + str(i+1), 
				summarize=cur_layer.nodes*cur_layer.prev_layer.nodes))

		if cur_layer.has_bias:
			check.append(tf.Print(cur_layer.biases, 
				[cur_layer.biases], message="biases " + str(i+1), summarize=cur_layer.nodes))

	sess.run(check)


def model_2(layers, iterator, process_input_output_fn, kwargs, task=TASK_TRAIN, loss=None):
	next_item = iterator.get_next()

	feature_batch, output_batch = process_input_output_fn(next_item[0], next_item[1], **kwargs)

	pre_calculations = []

	layers[0].features = feature_batch

	pre_calculations.append(layers[0].features)

	if loss is not None:
		loss.outputs = output_batch
		pre_calculations.append(loss.outputs)

	check = []

	for i, cur_layer in enumerate(layers[1:]):
		if cur_layer.has_weights:
			check.append(tf.Print(cur_layer.weights, 
				[cur_layer.weights], message="weights " + str(i+1), 
				summarize=cur_layer.nodes*cur_layer.prev_layer.nodes))

		if cur_layer.has_bias:
			check.append(tf.Print(cur_layer.biases, 
				[cur_layer.biases], message="biases " + str(i+1), summarize=cur_layer.nodes))

	with tf.control_dependencies([*pre_calculations, *check]):
		# calculate activations of the last layer which will recalculate activations of all the 
		# previous layers
		calc_activations_op = layers[-1].calc_activations()

	with tf.control_dependencies([calc_activations_op]):
		if task == TASK_PREDICT:
			return tf.argmax(calc_activations_op, 1)

		return loss.calculate_loss()


def gradient_descent_2(process_input_output_fn, kwargs, layers, loss, dataset, train_size, alpha, sess):

	n_iter = (train_size / BATCH_SIZE + (train_size % BATCH_SIZE != 0)) * NUMBER_EPOCHS_TO_TRAIN
	print("number of iterations gradient descent will perform:", n_iter)

	iterator = dataset.make_one_shot_iterator()

	optimizer = tf.train.GradientDescentOptimizer(alpha)

	cond = lambda i: tf.less(i, n_iter)

	def body(iter_n):
		operations = []
		prints = []

		calc_loss = model_2(layers, iterator, process_input_output_fn, kwargs, task=TASK_TRAIN, loss=loss)
		# train_op = optimizer.minimize(calc_loss)

		with tf.control_dependencies([calc_loss]):
			prints.append(tf.Print(calc_loss, [calc_loss], message="loss is "))

		with tf.control_dependencies([*prints]):
			return tf.add(iter_n, 1.0)


	while_loop = tf.while_loop(cond, body, [tf.constant(0.0)], parallel_iterations=1)
	sess.run(while_loop)

	check = []
	for i, cur_layer in enumerate(layers[1:]):
		if cur_layer.has_weights:
			check.append(tf.Print(cur_layer.weights, 
				[cur_layer.weights], message="weights " + str(i+1), 
				summarize=cur_layer.nodes*cur_layer.prev_layer.nodes))

		if cur_layer.has_bias:
			check.append(tf.Print(cur_layer.biases, 
				[cur_layer.biases], message="biases " + str(i+1), summarize=cur_layer.nodes))

	sess.run(check)


def fit(process_input_output_fn, kwargs, layers, loss, dataset, train_size, alpha=0.001):
	"""
	Fits the model to the dataset

	process_input_output_fn: function to process input and output
	kwargs: dictionary of arguments to be passed to above function
	layers: list of layers in the network
	loss: loss used for the network
	dataset: the dataset whose iterator returns the input and outputs in batches
	train_size: the size of the training data
	alpha: the step size
	"""
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# can perform weight initialization using Autoencoder

		gradient_descent_2(process_input_output_fn, kwargs, layers, 
			loss, dataset, train_size, alpha, sess)
		# gradient_descent(process_input_output_fn, kwargs, layers, 
		# 	loss, dataset, train_size, alpha, sess)

		saver.save(sess, 'save_1/')


def predict(process_input_output_fn, kwargs, layers, dataset, test_size, outputs):
	"""
	Predicts the output and computes accuracy

	process_input_output_fn: function to process input and output
	kwargs: dictionary of arguments to be passed to above function
	layers: list of layers in the network
	dataset: the dataset whose iterator returns the input and outputs in batches
	feature_columns: feature columns to get features from dataset
	test_size: size of the test dataset
	"""

	saver = tf.train.Saver()

	with tf.Session() as sess:

		iterator = dataset.make_one_shot_iterator()

		saver.restore(sess, 'save_1/')

		prediction_values = np.array([])
		while True:
			try:
				predictions = model(layers, iterator, process_input_output_fn, kwargs, 
					task=TASK_PREDICT)

				print_predictions = tf.Print(predictions, [predictions], 
					message="predictions are: ", summarize=test_size)
				pval = sess.run(print_predictions)
				prediction_values = np.append(prediction_values, pval)
			except:
				break

		print(str(prediction_values))
		print(outputs)

		print(metrics.accuracy_score(outputs, prediction_values))
