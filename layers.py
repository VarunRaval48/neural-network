from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import pandas as pd


class Layer(ABC):

	"""	
	nodes: the number of nodes in this layer
	is_input: True if this layer is an input layer
	name: the name of this layer
	"""

	def __init__(self, nodes, is_input=False, name=""):
		self.nodes = nodes
		self.is_input = is_input
		self.name = name

	@abstractmethod
	def calc_activations(self):
		pass

	@abstractmethod
	def get_activations(self):
		pass


class InputLayer(Layer):


	def __init__(self, nodes):
		Layer.__init__(self, nodes, True)

		# self.activations = tf.get_variable("input_activations",  
		# 	shape=[self.batch_size, self.nodes], initializer=tf.zeros_initializer())

		self.activations = None
		self.features = tf.placeholder(tf.float32)


	def calc_activations(self):
		"""
		Returns: the activations or the calculated features using feature columns
		"""
		self.activations = self.features
		return self.activations

	def get_activations(self):
		"""
		Returns: the activations or the calculated features using feature columns
		"""
		return self.activations


class HiddenLayer(Layer):

	def __init__(self, nodes, shape_weights, prev_layer, activation, grad_activation=None, name=""):
		"""
			nodes: number of nodes in this layer
			prev_layer: the layer before this layer (don't use this constructor for input layer)
			activation: the function used to perform activation of the neurons
			grad_activation: the function used to perform gradient of the activation function
		"""

		Layer.__init__(self, nodes, name=name)
		self.prev_layer = prev_layer
		self.activation = activation
		self.grad_activation = grad_activation

		prev_layer.next_layer = self

		# initializing next layer now so that we can identify last layer
		self.next_layer = None

		self.activations = None
		self.pre_activations = None
		self.grad_activation_weight = None
		self.grad_cost_pre_activation = None

		self.weights = tf.get_variable("weights", 
			shape=shape_weights, 
			initializer=tf.random_normal_initializer())



	def calc_activations(self):

		if self.activation is None:
			self.activations = self.calc_pre_activations()
		else:
			self.activations = self.activation(self.calc_pre_activations())

		return self.activations


	def get_pre_activations(self):
		return self.pre_activations

	def get_activations(self):
		return self.activations


	# remember that pre_activations, activations is an array or consist of many datapoints
	# calculate gradient of activations with respect to preactivations

	def calc_grad_activation_pre_activation(self):
		"""
		Returns: tensor with shape (batch size) x (number of activations)
		"""
		if self.grad_activation is None:
			return tf.ones(tf.shape(self.get_pre_activations()))
		else:
			return self.grad_activation(self.get_pre_activations())


	def get_weights(self):
		return self.weights


class FullyConnectedLayer(HiddenLayer):

	def __init__(self, nodes, prev_layer, activation, grad_activation=None, name=""):
		"""
			nodes: number of nodes in this layer
			prev_layer: the layer before this layer (don't use this constructor for input layer)
			activation: the function used to perform activation of the neurons
			grad_activation: the function used to perform gradient of the activation function
		"""

		shape_weights = [nodes, prev_layer.nodes]

		HiddenLayer.__init__(self, nodes, shape_weights, prev_layer, activation, grad_activation, name=name)

		self.biases = tf.get_variable("biases",
			shape=(nodes, 1), 
			initializer=tf.random_normal_initializer())


	def calc_pre_activations(self):

		self.pre_activations = tf.matmul(self.prev_layer.calc_activations(), 
			self.weights, transpose_b=True) + tf.transpose(self.biases)

		return self.pre_activations


	def calc_grad_cost_pre_activation_prev_layer(self):
		"""
		Returns: tensor of shape batch_size x previous layer activations
		"""
		if not self.prev_layer.is_input:
			# if self.next_layer is None:
			# 	cur_layer_grad_cost_pre_activation_fn = self.calc_grad_cost_pre_activation
			# else:
			# 	cur_layer_grad_cost_pre_activation_fn = self.next_layer.calc_grad_cost_pre_activation_prev_layer


			cur_layer_grad_cost_pre_activation_fn = self.next_layer.calc_grad_cost_pre_activation_prev_layer

			self.prev_layer.grad_cost_pre_activation = \
			tf.matmul(cur_layer_grad_cost_pre_activation_fn(), self.weights) * \
			self.prev_layer.calc_grad_activation_pre_activation()

			return self.prev_layer.grad_cost_pre_activation


	# have to do for multiple datapoints and multiple weights
	# for a datapoint, for all the output layer activations, it is same

	def calc_grad_activation_weight(self):
		"""
		Returns: tensor with shape (batch_size) x (previous layer units)
		"""
		# return tf.assign(self.grad_activation_weight, self.prev_layer.get_activations())

		self.grad_activation_weight = self.prev_layer.get_activations()

		return self.grad_activation_weight

	def get_grad_cost_pre_activation(self):
		return self.grad_cost_pre_activation

	def get_grad_activation_weight(self):
		return self.grad_activation_weight


	def calc_grad_cost_weight(self):
		"""
		Returns: tensor with shape same as weights of this layer
		"""
		# total = tf.zeros(shape=[self.nodes, self.prev_layer.nodes])

		# for i in range(self.batch_size):
		# 	total = tf.add(total, tf.matmul(self.get_grad_cost_pre_activation()[i:i+1, :], 
		# 		self.get_grad_activation_weight()[i:i+1, :], transpose_a=True))

		cond = lambda i, j: tf.less(i, tf.shape(self.get_grad_cost_pre_activation())[0])

		def body(i, total): 
			total = tf.add(total, tf.matmul(self.get_grad_cost_pre_activation()[i:i+1, :], 
				self.get_grad_activation_weight()[i:i+1, :], transpose_a=True))

			return tf.add(i, 1), total

		while_loop = tf.while_loop(cond, body, 
			[tf.constant(0), tf.zeros(shape=[self.nodes, self.prev_layer.nodes])])

		return while_loop[1]


	def calc_grad_cost_bias(self):
		"""
		Returns: tensor with shape same as bias of this layer
		"""
		return tf.transpose(tf.reduce_sum(self.get_grad_cost_pre_activation(), axis=0, keepdims=True))


class ConvolutionLayer(HiddenLayer):

	def __init__(self, shape_input, shape_filter, prev_layer, activation, grad_activation=None, name=""):
		"""
			feature_maps: number of nodes in this layer
			prev_layer: the layer before this layer (don't use this constructor for input layer)
			activation: the function used to perform activation of the neurons
			grad_activation: the function used to perform gradient of the activation function
		"""

		out_height = shape_input[1] - shape_filter[0] + 1
		out_width = shape_input[2] - shape_filter[0] + 1

		nodes = out_height * out_width * shape_filter[-1]

		HiddenLayer.__init__(self, nodes, shape_weights, prev_layer, activation, grad_activation, name=name)

		self.shape_input = shape_input
		self.shape_filter = shape_filter

		# self.biases = tf.get_variable("biases",
		# 	shape=(nodes, 1), 
		# 	initializer=tf.random_normal_initializer())


	def calc_pre_activations(self):

		self.pre_activations = tf.nn.conv2d(input=self.prev_layer.features, filter=self.weights,
											strides=[1, 1, 1, 1], padding='VALID')

		return self.pre_activations


	def calc_grad_cost_pre_activation_prev_layer(self):
		"""
		Returns: tensor of shape batch_size x previous layer activations
		"""
		if not self.prev_layer.is_input:
			cur_layer_grad_cost_pre_activation_fn = self.next_layer.calc_grad_cost_pre_activation_prev_layer

			self.prev_layer.grad_cost_pre_activation = \
			tf.nn.conv2d_backprop_input(input_sizes=self.shape_input, filter=self.weights, 
				out_backprop=cur_layer_grad_cost_pre_activation_fn(), strides=[1, 1, 1, 1], 
				padding='VALID') * \
			self.prev_layer.calc_grad_activation_pre_activation()

			return self.prev_layer.grad_cost_pre_activation


	# def calc_grad_activation_weight(self):
	# 	"""
	# 	Returns: tensor with shape (batch_size) x (previous layer units)

	# 	"""
	# 	# return tf.assign(self.grad_activation_weight, self.prev_layer.get_activations())

	# 	self.grad_activation_weight = self.prev_layer.get_activations()

	# 	return self.grad_activation_weight


	def get_grad_cost_pre_activation(self):
		return self.grad_cost_pre_activation


	def get_grad_activation_weight(self):
		return self.grad_activation_weight