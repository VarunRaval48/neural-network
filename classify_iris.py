import tensorflow as tf
import layers as l
from extras import *
from train import *


def model(layers, iterator, feature_columns, task=TASK_TRAIN):
	"""
	Creates the graph and returns values according to task

	layers: the list of initialized layers
	iterator: iterator over the dataset which gets items in batches
	feature_columns: list of feature columns to get features from dataset
	task: task for which this function is called

	Returns: a tensor of predictions if task is to predict 
			 a list of tensors of grad cost weights and grad cost biases if task is to train
	"""

	next_item = iterator.get_next()

	# converting next_item[0] to features
	feature_batch = tf.feature_column.input_layer(next_item[0], feature_columns)
	output_batch = tf.one_hot(next_item[1], depth=3, dtype=tf.int32)

	l.features = feature_batch
	l.outputs = output_batch

	prints = []
	check = []

	for i, cur_layer in enumerate(layers[1:]):
		check.append(tf.Print(cur_layer.weights, 
			[cur_layer.weights], message="weights " + str(i+1), 
			summarize=cur_layer.nodes*cur_layer.prev_layer.nodes))
		check.append(tf.Print(cur_layer.biases, 
			[cur_layer.biases], message="biases " + str(i+1), summarize=cur_layer.nodes))


	with tf.control_dependencies([l.features, l.outputs]):
		# calculate activations of the last layer which will recalculate activations of all the 
		# previous layers
		calc_activations_op = layers[-1].calc_activations()

	if task == TASK_PREDICT:
		return tf.argmax(calc_activations_op, 1)


	with tf.control_dependencies([*check, calc_activations_op]):
		# calculate error terms for the second layers which will calculate error terms for all 
		# the next layers
		calc_grad_cost_activation_op = layers[2].calc_grad_cost_activation_prev_layer()

		# calculate grad activation weight (not useful here)
		calc_grad_activation_weight_op = \
		[layers[i].calc_grad_activation_weight() for i in range(1, len(layers))]


	grad_cost_weights = []
	grad_cost_biases = []

	with tf.control_dependencies([*prints, calc_grad_cost_activation_op, 
		*calc_grad_activation_weight_op]):

		for i, cur_layer in enumerate(layers[1:]):
			grad_cost_weights.append(cur_layer.calc_grad_cost_weight())

			grad_cost_biases.append(cur_layer.calc_grad_cost_bias())

		return grad_cost_weights, grad_cost_biases


def make_layers(n_features):
	"""
	n_features: number of features in the dataset

	Returns: the list of layers
	"""

	hidden_layer_1_nodes = 10
	hidden_layer_2_nodes = 3
	n_classes = 3

	with tf.variable_scope("input_layer_scope"):
		input_layer = l.InputLayer(n_features)

	with tf.variable_scope("hidden_layer_1_scope"):
		hidden_layer_1 = l.HiddenLayer(hidden_layer_1_nodes, input_layer, 
											activation=None, name="hidden layer 1")

	with tf.variable_scope("hidden_layer_2_scope"):
		hidden_layer_2 = l.HiddenLayer(hidden_layer_2_nodes, hidden_layer_1, 
											activation=None, name="hidden layer 2")

	with tf.variable_scope("output_layer_scope"):
		output_layer = l.OutputLayer(n_classes, hidden_layer_2,  
											activation=None, grad_activation=None, 
											name="output layer") #TODO

	layers = [input_layer, hidden_layer_1, hidden_layer_2, output_layer]
	# layers = [input_layer, output_layer]

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

	alpha = 0.00001

	layers = make_layers(n_features)

	fit(model, layers, dataset, feature_columns, train_size, alpha=alpha)

	# print(test[1])
	print()
	predict(model, layers, test_dataset, feature_columns, test_size, test[1])