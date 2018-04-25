"""
This file shows how to use the layers defined in layers.py file, read dataset, and use
my implementation of gradient descent to perform training on a dataset.
"""

import tensorflow as tf
import layers as l
import loss as lo
from extras import *
from train import *

NUMBER_EPOCHS_TO_TRAIN = 1000
BATCH_SIZE = 120


def iris_process_input_output_fn(features, labels, feature_columns):
	"""
	features: output of iterator
	labels: output of iterator
	feature_columns: list of feature columns to get features from dataset

	Returns: processed feaures and labels
	"""
	feature_batch = tf.feature_column.input_layer(features, feature_columns)
	output_batch = tf.one_hot(labels, depth=3, dtype=tf.int32)

	return feature_batch, output_batch


def make_layers(n_features):
	"""
	n_features: number of features in the dataset

	Returns: the list of layers
	"""

	hidden_layer_1_nodes = [10]
	hidden_layer_2_nodes = [3]
	n_classes = [3]

	with tf.variable_scope("input_layer_scope"):
		input_layer = l.InputLayer([n_features])

	with tf.variable_scope("hidden_layer_1_scope"):
		hidden_layer_1 = l.FullyConnectedLayer(hidden_layer_1_nodes, input_layer, 
											activation=None, name="hidden layer 1")

	with tf.variable_scope("hidden_layer_2_scope"):
		hidden_layer_2 = l.FullyConnectedLayer(hidden_layer_2_nodes, hidden_layer_1, 
											activation=None, name="hidden layer 2")

	with tf.variable_scope("output_layer_scope"):
		hidden_layer_3 = l.FullyConnectedLayer(n_classes, hidden_layer_2, 
											activation=None, name="output layer")

	layers = [input_layer, hidden_layer_1, hidden_layer_2, hidden_layer_3]

	return layers


if __name__ == '__main__':

	batch_size = BATCH_SIZE

	train, test = load_data()
	dataset = train_input_fn(train[0], train[1], batch_size)

	test_size = len(test[0])
	test_dataset = eval_input_fn(test[0], test[1], test_size)

	keys = train[0].keys()
	feature_columns = []
	for feature_name in keys:
		feature_columns.append(tf.feature_column.numeric_column(key=feature_name))


	train_size = len(train[0])
	n_features = train[0].shape[1]

	print("size of the dataset is:", train_size)
	print("number of features are:", n_features)

	alpha = 0.01

	layers = make_layers(n_features)
	loss = lo.SoftmaxCrossEntropyLoss(layers[-1])

	kwargs = {}
	kwargs['feature_columns'] = feature_columns

	fit(iris_process_input_output_fn, kwargs, layers, loss, dataset, train_size, alpha)

	print()
	predict(iris_process_input_output_fn, kwargs, layers, test_dataset, test_size, test[1])