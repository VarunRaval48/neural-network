"""
This file contains various loss functions implemented by me
"""

from abc import ABC, abstractmethod
import tensorflow as tf


class Loss(ABC):

	"""
	A layer that is not actually a layer but is build upon a previous layer to calculate loss
	"""

	def __init__(self, prev_layer):
		self.prev_layer = prev_layer
		self.prev_layer.next_layer = self

		self.next_layer = None
		self.outputs = tf.placeholder(tf.int32)


	@abstractmethod
	def calc_grad_cost_pre_activation_prev_layer(self):
		pass


	@abstractmethod
	def calculate_loss(self):
		pass


class SoftmaxCrossEntropyLoss(Loss):
	"""
	calculates softmax cross entropy loss of the neural network
	"""

	def __init__(self, prev_layer):
		Loss.__init__(self, prev_layer)

		self.loss = None


	def calc_grad_cost_pre_activation_prev_layer(self):
		"""
		Returns: tensor of shape (batch_size) x (activations)
		"""

		self.prev_layer.grad_cost_pre_activation = \
		tf.subtract(tf.nn.softmax(self.prev_layer.get_activations()), tf.to_float(self.outputs)) * \
		self.prev_layer.calc_grad_activation_pre_activation()

		return self.prev_layer.grad_cost_pre_activation

	def calculate_loss(self):
		"""
		calculates the softmax cross entropy loss

		Returns: the tensor with scalar value of loss
		"""

		self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.outputs, 
					logits=self.prev_layer.get_activations()))

		return self.loss

	def get_loss(self):
		return self.loss