import tensorflow as tf
import numpy as np
import pandas as pd
from extras import *
import layers


NUMBER_EPOCHS_TO_TRAIN = 100
BATCH_SIZE = 120


# model is the list of initialized layers
# dataset is the dataset whose iterator returns the input and outputs in batches
# list of feature columns to get features from dataset
# train_size is the size of the dataset that each epoch should at least travel
# alpha is the step_size

def gradient_descent(model, dataset, feature_columns, train_size, alpha, sess):

	n_iter = (train_size / BATCH_SIZE + (train_size % BATCH_SIZE != 0)) * NUMBER_EPOCHS_TO_TRAIN
	print("number of iterations gradient descent will perform:", n_iter)

	iterator = dataset.make_one_shot_iterator().get_next()

	# run the gradient descent for number of epochs
	while n_iter:

		# do the following for an entire dataset

		# converting iterator[0] to features
		feature_batch = tf.feature_column.input_layer(iterator[0], feature_columns)
		output_batch = tf.one_hot(iterator[1], depth=3, dtype=tf.int32)

		operations = []
		prints = []

		# calculate activations of the last layer which will recalculate activations of all the 
		# previous layers

		check = []

		# for i, cur_layer in enumerate(model):
		# 	check.append(tf.Print(cur_layer.get_activations(), [cur_layer.get_activations()], message="activations before calc " + str(i), summarize=cur_layer.nodes))

		for i, cur_layer in enumerate(model[1:]):
			check.append(tf.Print(cur_layer.weights, [cur_layer.weights], message="weights " + str(i+1), summarize=cur_layer.nodes*cur_layer.prev_layer.nodes))
			check.append(tf.Print(cur_layer.biases, [cur_layer.biases], message="biases " + str(i+1), summarize=cur_layer.nodes))

		calc_activations_op = model[-1].calc_activations()

		# prints.append(tf.Print(cur_layer.get_grad_activation_weight(), [model[i].get_grad_activation_weight() for i in range(1, len(model))], message="grad activations weight before calc"))

		with tf.control_dependencies([*check, calc_activations_op]):
			# for i, cur_layer in enumerate(model):
			# 	prints.append(tf.Print(cur_layer.get_activations(), [cur_layer.get_activations()], message="activations after calc " + str(i), summarize=cur_layer.nodes))


			# calculate error terms for the second layers which will calculate error terms for all 
			# the next layers
			calc_grad_cost_activation_op = model[1].calc_grad_cost_activation()

			for i, cur_layer in enumerate(model[1:]):
				check.append(tf.Print(cur_layer.grad_cost_activation, [cur_layer.grad_cost_activation], message="grad_cost_activation " + str(i + 1), summarize=cur_layer.nodes))

			# calculate grad activation weight (not useful here)
			calc_grad_activation_weight_op = [model[i].calc_grad_activation_weight() for i in range(1, len(model))]

			for i, cur_layer in enumerate(model[1:]):
				check.append(tf.Print(cur_layer.grad_activation_weight, [cur_layer.grad_activation_weight], message="grad_activation_weight " + str(i + 1), summarize=cur_layer.prev_layer.nodes))


		# with tf.control_dependencies(calc_grad_activation_weight_op):
		# 	for i in range(1, len(model)):
				# prints.append(tf.Print(model[i].get_grad_activation_weight(), [model[i].get_grad_activation_weight()], message="grad activations weight after calc " + str(i)))


		with tf.control_dependencies([*check, calc_grad_cost_activation_op, 
			*calc_grad_activation_weight_op, calc_activations_op]):

			for i, cur_layer in enumerate(model[1:]):
				# print(cur_layer.name)
				# prints.append(tf.Print(cur_layer.get_activations(), [cur_layer.get_activations()], message="activations " + str(i)))

				# prints.append(tf.Print(cur_layer.get_grad_activation_weight(), [cur_layer.get_grad_activation_weight()], message="grad activations weight before weight update " + str(i)))

				grad_cost_weight = cur_layer.calc_grad_cost_weight()
				update = cur_layer.weights - alpha * (grad_cost_weight)
				operations.append(tf.assign(cur_layer.weights, update))

				prints.append(tf.Print(grad_cost_weight, [grad_cost_weight], message="grad_cost_weight " + str(i + 1), summarize=cur_layer.nodes * cur_layer.prev_layer.nodes))

				grad_cost_bias = cur_layer.calc_grad_cost_bias()
				update = cur_layer.biases - alpha * (grad_cost_bias)
				operations.append(tf.assign(cur_layer.biases, update))


		with tf.control_dependencies(operations):
			# prints.append(tf.Print(model[1].weights, [model[1].weights], message="hidden layer weights"))
			# prints.append(tf.Print(model[1].biases, [model[1].biases], message="hidden layer biases"))
			# prints.append(tf.Print(model[2].weights, [model[2].weights], message="output layer weights"))
			# prints.append(tf.Print(model[2].biases, [model[2].biases], message="output layer biases"))
			loss = model[-1].calculate_loss()

		to_print, loss = sess.run((prints, loss), feed_dict={layers.features : sess.run(feature_batch), 
			layers.outputs : sess.run(output_batch)})

		print('loss is:', loss)
		# print(sess.run(model[0].activations))
		# print(sess.run(model[1].activations))
		# print('weights are\n', sess.run(model[-1].weights))
		# print('biases are\n', sess.run(model[-1].biases))
		# print('weights are\n', sess.run(model[-2].weights))
		# print('biases are\n', sess.run(model[-2].biases))

		n_iter -= 1


# model is the list of initialized layers
# dataset is the dataset whose iterator returns the input and outputs in batches
# list of feature columns to get features from dataset
# train_size is the size of the training data
# alpha is the step size

def fit(model, dataset, feature_columns, train_size, alpha=0.001):

	# iterator = dataset.make_one_shot_iterator().get_next()

	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())

		# converting iterator[0] to features
		# feature_batch = tf.feature_column.input_layer(iterator[0], feature_columns)
		# output_batch = tf.one_hot(iterator[1], depth=3)

		# u, v, x, y, z = sess.run((model[-1].calc_activations(), model[1].calc_grad_cost_weight(), 
		# 	model[2].calc_grad_cost_weight(), model[1].calc_grad_cost_bias(), 
		# 	model[2].calc_grad_cost_bias()), 
		# feed_dict={layers.features : sess.run(feature_batch), 
		# 			layers.outputs : sess.run(output_batch)})

		# print(u, v, x, y, z)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# can perform weight initialization using Autoencoder

		gradient_descent(model, dataset, feature_columns, train_size, alpha, sess)


def model(n_features, batch_size):

	hidden_layer_1_nodes = 10
	n_classes = 3

	with tf.variable_scope("input_layer_scope"):
		input_layer = layers.InputLayer(n_features, batch_size)

	with tf.variable_scope("hidden_layer_1_scope"):
		hidden_layer_1 = layers.HiddenLayer(hidden_layer_1_nodes, batch_size, input_layer, 
											activation=None, name="hidden layer 1")

	with tf.variable_scope("output_layer_scope"):
		output_layer = layers.OutputLayer(n_classes, batch_size, hidden_layer_1,  
											activation=None, grad_activation=None, name="output layer") #TODO

	model = [input_layer, hidden_layer_1, output_layer]
	# model = [input_layer, output_layer]

	return model


if __name__ == '__main__':

	batch_size = BATCH_SIZE

	train, test = load_data()
	keys = train[0].keys()

	feature_columns = []
	for feature_name in keys:
		feature_columns.append(tf.feature_column.numeric_column(key=feature_name))


	dataset = train_input_fn(train[0], train[1], batch_size)

	train_size = len(train[0])
	n_features = train[0].shape[1]

	print("size of the dataset is:", train_size)
	print("number of features are:", n_features)

	model = model(n_features, batch_size)
	fit(model, dataset, feature_columns, train_size)

	# while True:
	# 	try:
	# 	except tf.errors.OutOfRangeError:
	# 		break
