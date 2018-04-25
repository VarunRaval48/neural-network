"""
This file contains methods to form architecture of neural network, add 
loss to it and perform a gradient descent step.
"""


from functools import reduce

import tensorflow as tf

NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False, 'Train the model using fp16')
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of images to process in a batch')

tf.app.flags.DEFINE_boolean('relu', False, 'Whether to use relu')
tf.app.flags.DEFINE_boolean('lrn', False, 'Whether to use lrn')
tf.app.flags.DEFINE_string('pool_type', 'avg', 'pool type to use')
tf.app.flags.DEFINE_boolean('t', False, 'Whether to train inner layers')


def _variable_with_weight_decay(name, shape, dtype=None, initializer=None, wd=None, trainable=True):

	var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, 
						  trainable=trainable)

	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)

	return var

def conv_1(images, trainable):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	in_channels = images.get_shape()[-1].value

	with tf.variable_scope('conv_1') as scope:
		kernel = _variable_with_weight_decay(name='weights', shape=[9, 9, in_channels, 64],
											 dtype=dtype,
											 initializer=\
											 	tf.truncated_normal_initializer(dtype=dtype),
											 trainable=trainable)

		conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='VALID')

		biases = tf.get_variable("biases", shape=[64], dtype=dtype, 
								 initializer=tf.truncated_normal_initializer(dtype=dtype),
								 trainable=trainable)

		conv1 = tf.tanh(tf.nn.bias_add(conv, biases))
		return conv1


def full(last_layer, n_classes):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	with tf.variable_scope('full_1') as scope:

		reshape = tf.reshape(last_layer, [FLAGS.batch_size, -1])
		# dim = reshape.get_shape()[1].value
		dim = reduce(lambda x, y: x * y, last_layer.get_shape().as_list()[1:], 1)

		weights = _variable_with_weight_decay(name='weights', shape=[dim, n_classes],
											  dtype=dtype,
											  initializer=\
											    tf.truncated_normal_initializer(dtype=dtype),
											  wd=0.001)

		biases = tf.get_variable("biases", shape=[n_classes], dtype=dtype, 
					   			 initializer=tf.truncated_normal_initializer(dtype=dtype))

		full1 = tf.add(tf.matmul(reshape, weights), biases)

	return full1


def inference(images, n_classes, relu, lrn, pool_type, trainable, stages=1):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

	last = conv_1(images, trainable)

	if relu:
		last = tf.nn.relu(last)

	if lrn:
		last = tf.nn.local_response_normalization(last)

	if pool_type == 'max':
		pool1 = tf.nn.max_pool(last, ksize=[1, 10, 10, 1], strides=[1, 5, 5, 1], padding='VALID')
	else:
		pool1 = tf.nn.avg_pool(last, ksize=[1, 10, 10, 1], strides=[1, 5, 5, 1], padding='VALID')


	full1 = full(pool1, n_classes)
	return full1


def loss(logits, labels):

	softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	mean_loss = tf.reduce_mean(softmax_loss)
	tf.add_to_collection('losses', mean_loss)

	return tf.add_n(tf.get_collection('losses'))


def train(loss, global_step, examples_per_epoch):

	num_batches_per_epoch = examples_per_epoch / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, 
									LEARNING_RATE_DECAY_FACTOR, staircase=True)

	opt = tf.train.GradientDescentOptimizer(lr)
	grads = opt.compute_gradients(loss)
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	with tf.control_dependencies([apply_gradient_op]):
		train_op = tf.no_op(name="train_op")

	return train_op