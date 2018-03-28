from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import pandas as pd


features = tf.placeholder(tf.float32)
outputs = tf.placeholder(tf.int32)


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


	def calc_activations(self):
		"""
		returns the activations or the calculated features using feature columns

		"""
		self.activations = features
		return self.activations

	def get_activations(self):
		"""
		returns the activations or the calculated features using feature columns

		"""
		return self.activations


class HiddenLayer(Layer):

	"""
		prev_layer: the layer before this layer (don't use this constructor for input layer)

		activation: the function used to perform activation of the neurons

		grad_activation: the function used to perform gradient of the activation function

	"""

	def __init__(self, nodes, prev_layer, activation, grad_activation=None, name=""):
		Layer.__init__(self, nodes, name=name)
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


		self.activations = None
		self.pre_activations = None
		self.grad_activation_weight = None
		self.grad_cost_activation = None


	def calc_pre_activations(self):

		self.pre_activations = tf.matmul(self.prev_layer.calc_activations(), 
			self.weights, transpose_b=True) + tf.transpose(self.biases)

		return self.pre_activations


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


	def calc_grad_cost_activation(self):
		"""
		Returns: tensor of shape batch_size x activations

		"""
		self.grad_cost_activation = \
		tf.matmul(self.next_layer.calc_grad_cost_activation(), self.next_layer.weights) * \
		self.calc_grad_activation_pre_activation()

		return self.grad_cost_activation


	# have to do for multiple datapoints and multiple weights
	# for a datapoint, for all the output layer activations, it is same

	def calc_grad_activation_weight(self):
		"""
		Returns: tensor with shape (batch_size) x (previous layer units)

		"""
		# return tf.assign(self.grad_activation_weight, self.prev_layer.get_activations())

		self.grad_activation_weight = self.prev_layer.get_activations()

		return self.grad_activation_weight

	def get_grad_cost_activation(self):
		return self.grad_cost_activation

	def get_grad_activation_weight(self):
		return self.grad_activation_weight


	def calc_grad_cost_weight(self):
		"""
		Returns: tensor with shape same as weights of this layer

		"""
		# total = tf.zeros(shape=[self.nodes, self.prev_layer.nodes])

		# for i in range(self.batch_size):
		# 	total = tf.add(total, tf.matmul(self.get_grad_cost_activation()[i:i+1, :], 
		# 		self.get_grad_activation_weight()[i:i+1, :], transpose_a=True))

		cond = lambda i, j: tf.less(i, tf.shape(self.get_grad_cost_activation())[0])

		def body(i, total): 
			total = tf.add(total, tf.matmul(self.get_grad_cost_activation()[i:i+1, :], 
				self.get_grad_activation_weight()[i:i+1, :], transpose_a=True))

			return tf.add(i, 1), total

		while_loop = tf.while_loop(cond, body, 
			[tf.constant(0), tf.zeros(shape=[self.nodes, self.prev_layer.nodes])])

		return while_loop[1]


	def calc_grad_cost_bias(self):
		"""
		Returns: tensor with shape same as bias of this layer

		"""
		return tf.transpose(tf.reduce_sum(self.get_grad_cost_activation(), axis=0, keepdims=True))


	def get_weights(self):
		return self.weights


	# remember that pre_activations, activations is an array or consist of many datapoints
	# calculate gradient of activations with respect to preactivations
	# here, activation function is equality i.e h(a) = a


	def calc_grad_activation_pre_activation(self):
		"""
		Returns: tensor with shape (batch size) x (number of activations)

		"""
		if self.grad_activation is None:
			return tf.ones(tf.shape(self.get_pre_activations()))
		else:
			# TODO replace it by tensors returned from methods
			return self.grad_activation(self.get_pre_activations())


class OutputLayer(HiddenLayer):
	"""
		output layer with softmax activation and logistic loss

	"""

	# have to do for multiple datapoints
	# outputs is a tensor of shape batch_size x output units


	def calc_grad_cost_activation(self):
		"""
		Returns: tensor of shape (batch_size) x (activations)

		"""
		self.grad_cost_activation = \
		tf.subtract(tf.nn.softmax(self.get_activations()), tf.to_float(outputs)) * \
		self.calc_grad_activation_pre_activation()

		return self.grad_cost_activation


	def calculate_loss(self):
		"""
		calculates the softmax cross entropy loss

		Returns: the tensor with scalar value of loss
		
		"""
		return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs, 
					logits=self.get_activations()))
