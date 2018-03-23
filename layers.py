from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import pandas as pd


features = tf.placeholder(tf.float32)
outputs = tf.placeholder(tf.int32)


class Layer(ABC):

	def __init__(self, nodes, batch_size, is_input=False, name=""):
		"""	
		nodes is the number of nodes in this layer

		"""
		self.nodes = nodes
		self.batch_size = batch_size
		self.is_input = is_input
		self.name = name

	# @abstractmethod
	# def get_activations(self):
	# 	pass


class InputLayer(Layer):

	def __init__(self, nodes, batch_size):
		Layer.__init__(self, nodes, batch_size, True)

		self.activations = tf.get_variable("input_activations",  
			shape=[self.batch_size, self.nodes], initializer=tf.zeros_initializer())


	"""
	features is the tensor which is the features to feed forward

	returns the activations or the calculated features using feature columns

	"""

	def calc_activations(self):
		return tf.assign(self.activations, features)

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
	def __init__(self, nodes, batch_size, prev_layer, activation, grad_activation=None, name=""):
		Layer.__init__(self, nodes, batch_size, name=name)
		self.prev_layer = prev_layer
		self.activation = activation
		self.grad_activation = grad_activation

		prev_layer.next_layer = self

		self.weights = tf.get_variable("weights", 
			shape=(nodes, prev_layer.nodes), 
			initializer=tf.random_normal_initializer())

		self.biases = tf.get_variable("biases",
			shape=(nodes, 1), 
			initializer=tf.random_normal_initializer())

		self.pre_activations = tf.get_variable("pre_activations", 
			shape=[self.batch_size, self.nodes], 
			initializer=tf.zeros_initializer())

		self.activations = tf.get_variable("activations", 
			shape=[self.batch_size, self.nodes], 
			initializer=tf.zeros_initializer())

		self.grad_cost_activation = tf.get_variable("grad_cost_activation", 
			shape=[self.batch_size, self.nodes], 
			initializer=tf.zeros_initializer())

		self.grad_activation_weight = tf.get_variable("grad_activation_weight", 
			shape=[self.batch_size, self.prev_layer.nodes], 
			initializer=tf.zeros_initializer())


	def calc_pre_activations(self):
		return tf.assign(self.pre_activations, tf.matmul(self.prev_layer.calc_activations(), 
			self.weights, transpose_b=True) + tf.transpose(self.biases))


	def calc_activations(self):
		if self.activation is None:
			return tf.assign(self.activations, self.calc_pre_activations())
		else:
			return tf.assign(self.activations, self.activation(self.calc_pre_activations()))


	def get_pre_activations(self):
		return self.pre_activations

	def get_activations(self):
		return self.activations


	# def get_activations(self):
	# 	return self.activations

	# returns tensor of shape batch_size x activations
	def calc_grad_cost_activation(self):
		return tf.assign(self.grad_cost_activation, 
			tf.matmul(self.next_layer.calc_grad_cost_activation(), self.next_layer.weights) * \
			self.calc_grad_activation_pre_activation())


	# have to do for multiple datapoints and multiple weights
	# for a datapoint, for all the output layer activations, it is same
	# REMOVE outputs is a tensor containing activation values of previous layer
	# shape is batch_size x input units
	def calc_grad_activation_weight(self):
		return tf.assign(self.grad_activation_weight, self.prev_layer.get_activations())

	def get_grad_cost_activation(self):
		return self.grad_cost_activation

	def get_grad_activation_weight(self):
		return self.grad_activation_weight


	# REMOVE outputs is a tensor which contains addition of gradients over entire batch size
	def calc_grad_cost_weight(self):
		total = tf.zeros(shape=[self.nodes, self.prev_layer.nodes])

		for i in range(self.batch_size):
			total = tf.add(total, tf.matmul(self.get_grad_cost_activation()[i:i+1, :], 
				self.get_grad_activation_weight()[i:i+1, :], transpose_a=True))

		return total

	def calc_grad_cost_bias(self):
		return tf.transpose(tf.reduce_sum(self.get_grad_cost_activation(), axis=0, keepdims=True))


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
			return self.grad_activation(self.get_pre_activations())



# output layer with softmax activation and logistic loss

class OutputLayer(HiddenLayer):
	"""
		prev_layer is the layer before this layer (don't use this constructor for input layer)

		activation is the function used to perform activation of the neurons

	"""
	# def __init__(self, nodes, batch_size, prev_layer, activation, grad_activation=None):
	# 	HiddenLayer.__init__(self, nodes, batch_size, prev_layer, activation, grad_activation)


	# have to do for multiple datapoints
	# outputs is a tensor of shape batch_size x output units
	# TODO do following such that outputs is not one hot tensor
	def calc_grad_cost_activation(self):
		return tf.assign(self.grad_cost_activation, tf.subtract(tf.nn.softmax(self.get_activations()), tf.to_float(outputs)) * \
				self.calc_grad_activation_pre_activation())

	def calculate_loss(self):
		return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs, 
					logits=self.get_activations()))
