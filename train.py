import tensorflow as tf
import numpy as np
import pandas as pd
import layers as l
from sklearn import metrics


NUMBER_EPOCHS_TO_TRAIN = 1000
BATCH_SIZE = 120

TASK_TRAIN = 'train'
TASK_PREDICT = 'predict'


def gradient_descent(model, layers, loss, dataset, feature_columns, train_size, alpha, sess):
	"""
	Perform the gradient descent and updates weights, biases

	model: function that is used to calculate grad_cost weights and biases for update
	layers: the list of initialized layers
	dataset: the dataset whose iterator returns the input and outputs in batches
	feature_columns: list of feature columns to get features from dataset
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

		grad_cost_weights, grad_cost_biases = model(layers, iterator, feature_columns, task=TASK_TRAIN)

		for i, cur_layer in enumerate(layers[1:]):
			update = tf.subtract(cur_layer.weights, (alpha / BATCH_SIZE) * (grad_cost_weights[i]))
			operations.append(tf.assign(cur_layer.weights, update))

			update = tf.subtract(cur_layer.biases, (alpha / BATCH_SIZE) * (grad_cost_biases[i]))
			operations.append(tf.assign(cur_layer.biases, update))

		with tf.control_dependencies(operations):
			calc_loss = loss.calculate_loss()
			prints.append(tf.Print(calc_loss, [calc_loss], message="loss is "))

		with tf.control_dependencies([*prints, calc_loss]):
			return tf.add(iter_n, 1.0)


	while_loop = tf.while_loop(cond, body, [tf.constant(0.0)], parallel_iterations=1)
	sess.run(while_loop)

	check = []
	for i, cur_layer in enumerate(layers[1:]):
		check.append(tf.Print(cur_layer.weights, 
			[cur_layer.weights], message="weights " + str(i+1), 
			summarize=cur_layer.nodes*cur_layer.prev_layer.nodes))
		check.append(tf.Print(cur_layer.biases, 
			[cur_layer.biases], message="biases " + str(i+1), summarize=cur_layer.nodes))

	sess.run(check)

	# to_print, loss = sess.run((prints, loss), 
	# 	feed_dict={layers.features : sess.run(feature_batch), 
	# 	layers.outputs : sess.run(output_batch)})


def fit(model, layers, loss, dataset, feature_columns, train_size, alpha=0.001):
	"""
	Fits the model to the dataset

	model: function that is used to calculate grad_cost weights and biases for update
	layers: list of layers in the network
	dataset: the dataset whose iterator returns the input and outputs in batches
	feature_columns: feature columns to get features from dataset
	train_size: the size of the training data
	alpha: the step size
	"""
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# can perform weight initialization using Autoencoder

		gradient_descent(model, layers, loss, dataset, feature_columns, train_size, alpha, sess)

		saver.save(sess, 'save_1/')


def predict(model, layers, dataset, feature_columns, test_size, outputs):
	"""
	Predicts the output and computes accuracy


	model: function that is used to calculate grad_cost weights and biases for update
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
				predictions = model(layers, iterator, feature_columns, task=TASK_PREDICT)

				print_predictions = tf.Print(predictions, [predictions], 
					message="predictions are: ", summarize=test_size)
				_, pval = sess.run((print_predictions, predictions))
				prediction_values = np.append(prediction_values, pval)
			except:
				break

		print(str(prediction_values))
		print(outputs)

		print(metrics.accuracy_score(outputs, prediction_values))
		