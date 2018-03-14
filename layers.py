from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import pandas as pd
from extras import *

class Layer(ABC):

	def __init__(self, nodes, batch_size, is_input=False):
		"""	
		nodes is the number of nodes in this layer

		"""
		self.nodes = nodes
		self.batch_size = batch_size
		self.is_input = is_input


	@abstractmethod
	def get_activations(self):
		pass


class InputLayer(Layer):

	def __init__(self, nodes, batch_size):
		Layer.__init__(self, nodes, batch_size, True)

		self.activations = tf.get_variable("input_activations",  
			shape=[self.batch_size, self.nodes], initializer=tf.zeros_initializer())


	"""
	features is the tensor which is the features to feed forward

	returns the activations or the calculated features using feature columns

	"""

	def calc_activations(self, features):
		assignment = tf.assign(self.activations, tf.reshape(features, [-1, self.nodes]))
		return assignment

	"""
	returns the activations or the calculated features using feature columns

	"""

	def get_activations(self):
		return self.activations


class HiddenLayer(Layer):

	"""
		prev_layer is the layer before this layer (don't use this constructor for input layer)

		activation is the function used to perform activation of the neurons

	"""
	def __init__(self, nodes, batch_size, prev_layer, activation, grad_activation=None):
		Layer.__init__(self, nodes, batch_size)
		self.prev_layer = prev_layer
		self.activation = activation
		self.grad_activation = None

		prev_layer.next_layer = self

		self.weights = tf.get_variable("weights", 
			shape=(nodes, prev_layer.nodes), 
			initializer=tf.random_normal_initializer())

		self.biases = tf.get_variable("biases",
			shape=(nodes), 
			initializer=tf.random_normal_initializer())

		self.pre_activations = tf.get_variable("pre_activations", 
			shape=[self.batch_size, self.nodes], 
			initializer=tf.zeros_initializer())

		self.activations = tf.get_variable("activations", 
			shape=[self.batch_size, self.nodes], 
			initializer=tf.zeros_initializer())


	def calc_pre_activations(self):
		if self.prev_layer.is_input:
			assignment = tf.assign(self.pre_activations, 
									tf.matmul(self.prev_layer.get_activations(), 
										self.weights, transpose_b=True))
		else:
			assignment = tf.assign(self.pre_activations, 
									tf.matmul(self.prev_layer.calc_activations(), 
										self.weights, transpose_b=True))

		return assignment


	def calc_activations(self):
		if self.activation is None:
			assignment = tf.assign(self.activations, self.calc_pre_activations())
		else:
			assignment = tf.assign(self.activations, self.activation(self.calc_pre_activations()))

		return assignment


	def get_activations(self):
		return self.activations


	def _calc_grad_cost_activation(self):
		return tf.matmul(self.next_layer._calc_grad_cost_activation(), self.next_layer.weights) * \
				self.calc_grad_activation_pre_activation()


	# have to do for multiple datapoints and multiple weights
	# for a datapoint, for all the output layer activations, it is same
	# REMOVE outputs is a tensor containing activation values of previous layer
	# shape is batch_size x input units
	def _calc_grad_activation_weight(self):
		return self.prev_layer.get_activations()


	# REMOVE outputs is a tensor which contains addition of gradients over entire batch size
	def calc_grad_cost_weight(self):
		total = tf.zeros(shape=[self.nodes, self.prev_layer.nodes])

		for i in range(self.batch_size):
			total += tf.matmul(self._calc_grad_cost_activation()[i:i+1, :], 
					self._calc_grad_activation_weight()[i:i+1, :], transpose_a=True)

		return total


	def get_weights(self):
		return self.weights


	# remember that pre_activations, activations is an array or consist of many datapoints
	# calculate gradient of activations with respect to preactivations
	# here, activation function is equality i.e h(a) = a
	def calc_grad_activation_pre_activation(self):
		if self.grad_activation is None:
			return tf.ones([self.batch_size, self.nodes])
		else:
			# TODO replace it by tensors returned from methods
			return grad_activation(self.pre_activations)



class OutputLayer(HiddenLayer):
	"""
		prev_layer is the layer before this layer (don't use this constructor for input layer)

		activation is the function used to perform activation of the neurons

	"""
	# def __init__(self, nodes, batch_size, prev_layer, activation, grad_activation=None):
	# 	HiddenLayer.__init__(self, nodes, batch_size, prev_layer, activation, grad_activation)


	# have to do for multiple datapoints
	# outputs is a tensor of shape batch_size x output units
	def _calc_grad_cost_activation(self):
		return (self.get_activations() - outputs) * self.calc_grad_activation_pre_activation()



def model(n_features, batch_size):

	hidden_layer_1_nodes = 10
	n_classes = 3

	with tf.variable_scope("input_layer_scope"):
		input_layer = InputLayer(n_features, batch_size)

	with tf.variable_scope("hidden_layer_1_scope"):
		hidden_layer_1 = HiddenLayer(hidden_layer_1_nodes, batch_size, input_layer, activation=None)

	with tf.variable_scope("output_layer_scope"):
		output_layer = OutputLayer(n_classes, batch_size, hidden_layer_1, activation=tf.nn.softmax, 
									grad_activation=None) #TODO

	layers = [input_layer, hidden_layer_1, output_layer]

	return layers

outputs = tf.placeholder(tf.float32)

# layers is the list of initialized layers
# dataset is the dataset whose iterator returns the input and outputs in batches
# list of feature columns to get features from dataset

def fit(layers, dataset, feature_columns):
	iterator = dataset.make_one_shot_iterator().get_next()

	sess.run(tf.global_variables_initializer())

	# converting iterator[0] to features
	feature_batch = tf.feature_column.input_layer(iterator[0], feature_columns)
	output_batch = tf.one_hot(iterator[1], depth=3)

	# TODO implement using placeholder
	input_layer_activations = sess.run(layers[0].calc_activations(feature_batch))

	sess.run(layers[-1].calc_activations())

	sess.run((layers[1].calc_grad_cost_weight(), layers[2].calc_grad_cost_weight()), 
		feed_dict={outputs : sess.run(output_batch)})



if __name__ == '__main__':
	
	with tf.Session() as sess:
		batch_size = 10

		train, test = load_data()
		keys = train[0].keys()

		feature_columns = []
		for feature_name in keys:
			feature_columns.append(tf.feature_column.numeric_column(key=feature_name))


		dataset = train_input_fn(train[0], train[1], batch_size)

		n_features = train[0].shape[1]

		layers = model(n_features, batch_size)
		fit(layers, dataset, feature_columns)

		# while True:
		# 	try:
		# 	except tf.errors.OutOfRangeError:
		# 		break
