import tensorflow as tf
import numpy as np
import pandas as pd
from extras import *
import layers


NUMBER_EPOCHS_TO_TRAIN = 1000
BATCH_SIZE = 120


def gradient_descent(model, dataset, feature_columns, train_size, alpha, sess):
	"""
	model: the list of initialized layers
	dataset: the dataset whose iterator returns the input and outputs in batches
	feature_columns: list of feature columns to get features from dataset
	train_size: the size of the dataset that each epoch should at least travel
	alpha: the step_size

	Calculates the gradient descent and perform weight updates
	"""

	# run the gradient descent on entire dataset for number of epochs

	n_iter = (train_size / BATCH_SIZE + (train_size % BATCH_SIZE != 0)) * NUMBER_EPOCHS_TO_TRAIN
	print("number of iterations gradient descent will perform:", n_iter)

	iterator = dataset.make_one_shot_iterator()

	cond = lambda i: tf.less(i, n_iter)

	def body(iter_n):
		next_item = iterator.get_next()

		# converting next_item[0] to features
		feature_batch = tf.feature_column.input_layer(next_item[0], feature_columns)
		output_batch = tf.one_hot(next_item[1], depth=3, dtype=tf.int32)

		layers.features = feature_batch
		layers.outputs = output_batch

		operations = []
		prints = []
		check = []

		for i, cur_layer in enumerate(model[1:]):
			check.append(tf.Print(cur_layer.weights, 
				[cur_layer.weights], message="weights " + str(i+1), 
				summarize=cur_layer.nodes*cur_layer.prev_layer.nodes))
			check.append(tf.Print(cur_layer.biases, 
				[cur_layer.biases], message="biases " + str(i+1), summarize=cur_layer.nodes))


		with tf.control_dependencies([layers.features, layers.outputs]):
			# calculate activations of the last layer which will recalculate activations of all the 
			# previous layers
			calc_activations_op = model[-1].calc_activations()

		with tf.control_dependencies([*check, calc_activations_op]):
			# calculate error terms for the second layers which will calculate error terms for all 
			# the next layers
			calc_grad_cost_activation_op = model[1].calc_grad_cost_activation()

			# calculate grad activation weight (not useful here)
			calc_grad_activation_weight_op = \
			[model[i].calc_grad_activation_weight() for i in range(1, len(model))]


		with tf.control_dependencies([*prints, calc_grad_cost_activation_op, 
			*calc_grad_activation_weight_op]):

			for i, cur_layer in enumerate(model[1:]):
				grad_cost_weight = cur_layer.calc_grad_cost_weight()
				update = cur_layer.weights - alpha * (grad_cost_weight)
				operations.append(tf.assign(cur_layer.weights, update))

				grad_cost_bias = cur_layer.calc_grad_cost_bias()
				update = cur_layer.biases - alpha * (grad_cost_bias)
				operations.append(tf.assign(cur_layer.biases, update))

		with tf.control_dependencies(operations):
			loss = model[-1].calculate_loss()
			prints.append(tf.Print(loss, [loss], message="loss is "))

		with tf.control_dependencies([*prints, loss]):
			return tf.add(iter_n, 1.0)


	while_loop = tf.while_loop(cond, body, [tf.constant(0.0)], parallel_iterations=1)
	sess.run(while_loop)

	# to_print, loss = sess.run((prints, loss), 
	# 	feed_dict={layers.features : sess.run(feature_batch), 
	# 	layers.outputs : sess.run(output_batch)})


def fit(model, dataset, feature_columns, train_size, alpha=0.001):
	"""
	model: the list of initialized layers
	dataset: the dataset whose iterator returns the input and outputs in batches
	list: feature columns to get features from dataset
	train_size: the size of the training data
	alpha: the step size

	Fits the model to the dataset

	"""

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# can perform weight initialization using Autoencoder

		gradient_descent(model, dataset, feature_columns, train_size, alpha, sess)


def model(n_features):
	"""
	n_features: number of features in the dataset

	Returns: the list of layers

	"""

	hidden_layer_1_nodes = 10
	hidden_layer_2_nodes = 3
	n_classes = 3

	with tf.variable_scope("input_layer_scope"):
		input_layer = layers.InputLayer(n_features)

	with tf.variable_scope("hidden_layer_1_scope"):
		hidden_layer_1 = layers.HiddenLayer(hidden_layer_1_nodes, input_layer, 
											activation=None, name="hidden layer 1")

	with tf.variable_scope("hidden_layer_2_scope"):
		hidden_layer_2 = layers.HiddenLayer(hidden_layer_2_nodes, hidden_layer_1, 
											activation=None, name="hidden layer 2")

	with tf.variable_scope("output_layer_scope"):
		output_layer = layers.OutputLayer(n_classes, hidden_layer_2,  
											activation=None, grad_activation=None, 
											name="output layer") #TODO

	model = [input_layer, hidden_layer_1, hidden_layer_2, output_layer]
	# model = [input_layer, output_layer]

	return model


if __name__ == '__main__':

	batch_size = BATCH_SIZE

	train, test = load_data()
	dataset = train_input_fn(train[0], train[1], batch_size)

	keys = train[0].keys()
	feature_columns = []
	for feature_name in keys:
		feature_columns.append(tf.feature_column.numeric_column(key=feature_name))


	train_size = len(train[0])
	n_features = train[0].shape[1]

	print("size of the dataset is:", train_size)
	print("number of features are:", n_features)

	alpha = 0.00001

	model = model(n_features)
	fit(model, dataset, feature_columns, train_size, alpha=alpha)
