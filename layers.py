"""
This file contains various layers implemented by me.
"""

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

	def __init__(self, shape_nodes, prev_layer=None, is_input=False, name=""):
		self.shape_nodes = shape_nodes
		self.nodes = np.prod(shape_nodes)
		self.is_input = is_input
		self.name = name

		self.has_weights = False
		self.has_bias = False

		if prev_layer is not None:
			prev_layer.next_layer = self
			self.prev_layer = prev_layer

		# initializing next layer now so that we can identify last layer
		self.next_layer = None

		self.require_loss = False


	@abstractmethod
	def calc_activations(self):
		pass

	@abstractmethod
	def get_activations(self):
		pass


class InputLayer(Layer):


	def __init__(self, shape_nodes):
		Layer.__init__(self, shape_nodes, is_input=True)

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

	def __init__(self, shape_nodes, shape_weights, prev_layer, activation, 
		grad_activation=None, name=""):
		"""
			nodes: number of nodes in this layer
			prev_layer: the layer before this layer (don't use this constructor for input layer)
			activation: the function used to perform activation of the neurons
			grad_activation: the function used to perform gradient of the activation function
		"""

		Layer.__init__(self, shape_nodes, prev_layer=prev_layer, name=name)
		self.activation = activation
		self.grad_activation = grad_activation

		self.activations = None
		self.pre_activations = None
		self.grad_activation_weight = None
		self.grad_cost_pre_activation = None

		self.weights = tf.get_variable("weights", 
			shape=shape_weights, 
			initializer=tf.random_normal_initializer())

		self.has_weights = True



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


	def get_weights(self):
		return self.weights


def flatten_except_first(inp):
	temp_shape = tf.shape(inp)
	return tf.reshape(inp, [temp_shape[0], tf.cumprod(temp_shape[1:])[-1]])


class FullyConnectedLayer(HiddenLayer):

	def __init__(self, shape_nodes, prev_layer, activation, grad_activation=None, name=""):
		"""
			nodes: number of nodes in this layer
			prev_layer: the layer before this layer (don't use this constructor for input layer)
			activation: the function used to perform activation of the neurons
			grad_activation: the function used to perform gradient of the activation function
		"""

		shape_weights = [shape_nodes[0], prev_layer.nodes]

		HiddenLayer.__init__(self, shape_nodes, shape_weights, prev_layer, activation, 
			grad_activation, name=name)

		self.biases = tf.get_variable("biases",
			shape=(shape_nodes[0], 1), 
			initializer=tf.random_normal_initializer())

		self.has_bias = True


	def calc_pre_activations(self):

		prev_layer_activations = self.prev_layer.calc_activations()

		# flatten here and repack during grad cost pre activation
		# also flatten during grad_cost_pre_activation (grad_activation_pre_activation)
		# and during grad_activation_weight

		# retain 1st dimension and flatten rest
		prev_layer_activations = tf.cond(tf.not_equal(tf.size(self.prev_layer.shape_nodes), 1), 
			lambda: flatten_except_first(prev_layer_activations), lambda: prev_layer_activations)

		self.pre_activations = tf.add(tf.matmul(prev_layer_activations, 
					self.weights, transpose_b=True), tf.transpose(self.biases))

		return self.pre_activations


	def calc_grad_cost_pre_activation_prev_layer(self):
		"""
		Returns: tensor of shape batch_size x previous layer activations
		"""
		if not self.prev_layer.is_input:
			cur_layer_grad_cost_pre_activation_fn = \
			self.next_layer.calc_grad_cost_pre_activation_prev_layer

			prev_layer_grad_act_pre_act = self.prev_layer.calc_grad_activation_pre_activation()
			prev_layer_grad_act_pre_act = \
			tf.cond(tf.not_equal(tf.size(self.prev_layer.shape_nodes), 1), 
				lambda: flatten_except_first(prev_layer_grad_act_pre_act), 
				lambda: prev_layer_grad_act_pre_act)

			prev_layer_grad_cost_pre_activation = \
			tf.multiply(tf.matmul(cur_layer_grad_cost_pre_activation_fn(), self.weights),
				prev_layer_grad_act_pre_act)

			self.prev_layer.grad_cost_pre_activation = \
			tf.cond(tf.not_equal(tf.size(self.prev_layer.shape_nodes), 1), 
				lambda: tf.reshape(prev_layer_grad_cost_pre_activation, 
					tf.shape(self.prev_layer.get_pre_activations())), 
				lambda: prev_layer_grad_cost_pre_activation)

			return self.prev_layer.grad_cost_pre_activation

	def calc_grad_activation_weight(self):
		"""
		Returns: tensor with shape (batch_size) x (previous layer units)
		"""
		# return tf.assign(self.grad_activation_weight, self.prev_layer.get_activations())

		grad_activation_weight = self.prev_layer.get_activations()

		self.grad_activation_weight = tf.cond(tf.not_equal(tf.size(self.prev_layer.shape_nodes), 1), 
			lambda: flatten_except_first(grad_activation_weight), lambda: grad_activation_weight)

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
		return tf.transpose(tf.reduce_sum(self.get_grad_cost_pre_activation(), axis=0, 
			keepdims=True))


class ConvolutionLayer(HiddenLayer):

	def __init__(self, shape_filter, strides, prev_layer, activation, grad_activation=None, 
		name=""):
		"""
			shape_filter: [1, height, width, 1]
			prev_layer: the layer before this layer (don't use this constructor for input layer)
			activation: the function used to perform activation of the neurons
			grad_activation: the function used to perform gradient of the activation function
		"""

		# TODO check
		out_height = ((prev_layer.shape_nodes[0] - shape_filter[0]) / strides[1]) + 1
		out_width = ((prev_layer.shape_nodes[1] - shape_filter[1]) / strides[2]) + 1
		out_channels = shape_filter[-1]

		shape_nodes = [out_height, out_width, out_channels]

		shape_weights = shape_filter

		HiddenLayer.__init__(self, shape_nodes, shape_weights, prev_layer, activation, 
			grad_activation, name=name)

		self.shape_filter = shape_filter

		# shape_bias = [shape_filter[0], shape_filter[1], ]

		# self.biases = tf.get_variable("biases",
		# 	shape=(nodes, 1), 
		# 	initializer=tf.random_normal_initializer())


	def calc_pre_activations(self):

		self.pre_activations = tf.nn.conv2d(input=self.prev_layer.calc_activations(), 
			filter=self.weights, strides=[1, 1, 1, 1], padding='VALID')

		self.shape_input = tf.shape(self.prev_layer.get_activations())
		return self.pre_activations


	def calc_grad_cost_pre_activation_prev_layer(self):
		"""
		Returns: tensor of shape batch_size x previous layer activations
		"""
		if not self.prev_layer.is_input:
			cur_layer_grad_cost_pre_activation_fn = \
			self.next_layer.calc_grad_cost_pre_activation_prev_layer

			self.prev_layer.grad_cost_pre_activation = \
			tf.multiply(tf.nn.conv2d_backprop_input(input_sizes=self.shape_input, 
				filter=self.weights, out_backprop=cur_layer_grad_cost_pre_activation_fn(), 
				strides=[1, 1, 1, 1], padding='VALID'),
			self.prev_layer.calc_grad_activation_pre_activation())

			return self.prev_layer.grad_cost_pre_activation


	def calc_grad_cost_weight(self):
		"""
		Returns: tensor with shape same as weights of this layer
		"""
		return tf.nn.conv2d_backprop_filter(input=self.prev_layer.get_activations(), 
			filter_sizes=self.shape_filter, out_backprop=self.get_grad_cost_pre_activation(),
			strides=[1, 1, 1, 1], padding='VALID')


class PoolingLayer(Layer):

	def __init__(self, prev_layer, k_size, strides):
		"""
		prev_layer: previous layer
		k_size: kernel size with which to perform max pooling
		"""

		out_height = ((prev_layer.shape_nodes[0] - k_size[1]) / strides[1]) + 1
		out_width = ((prev_layer.shape_nodes[1] - k_size[2]) / strides[2]) + 1
		out_channels = prev_layer.shape_nodes[-1]
		shape_nodes = [out_width, out_height, out_channels]

		Layer.__init__(self, shape_nodes, prev_layer=prev_layer)

		self.k_size = k_size
		self.strides = strides

		self.require_loss = True
		self.loss = None


	def calc_pre_activations(self):
		self.pre_activations = tf.nn.max_pool(value=self.prev_layer.calc_activations(), 
			ksize=self.k_size, strides=self.strides, padding="VALID")

		self.shape_input = tf.shape(self.prev_layer.get_activations())
		return self.pre_activations

	def get_pre_activations(self):
		return self.pre_activations

	def calc_activations(self):
		self.activations = self.calc_pre_activations()
		return self.activations

	def get_activations(self):
		return self.activations


	def calc_grad_cost_pre_activation_prev_layer(self):
		"""
		Returns: tensor of shape batch_size x previous layer activations
		"""
		temp_depend = self.next_layer.calc_grad_cost_pre_activation_prev_layer()

		with tf.control_dependencies([temp_depend]):
			prev_layer_pre_act = self.prev_layer.get_pre_activations()
			prev_layer_pre_act_f = tf.reshape(prev_layer_pre_act, [-1])

			# it = np.nditer(np.ones())
			# while not it.finished

			prev_layer_grad_cost_pre_activation = \
			tf.map_fn(lambda x: tf.gradients(ys=self.loss, xs=x)[0], prev_layer_pre_act_f)

			# prev_layer_grad_cost_pre_activation = tf.gradients(self.loss, prev_layer_pre_act_f)

			self.prev_layer.grad_cost_pre_activation = \
			tf.reshape(prev_layer_grad_cost_pre_activation, tf.shape(prev_layer_pre_act))

			return self.prev_layer.grad_cost_pre_activation


	def calc_grad_activation_pre_activation(self):
		return tf.ones(tf.shape(self.get_pre_activations()))