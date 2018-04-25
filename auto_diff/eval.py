"""
This file contains methods to evaluate the trained model
"""
from read_args import read_command

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score

import model, read_input

tf.app.flags.DEFINE_integer('e', 1, 'eval measure to use')
tf.app.flags.DEFINE_string('c', '', 'directory to store checkpoints')
tf.app.flags.DEFINE_boolean('no-train', True, 'whether to train the inside weights')

def eval_measure_1(sess, logits, labels, total_examples):
	true_count = 0

	top_k_op = tf.nn.in_top_k(logits, labels, 1)

	iter_n = 0
	while True:
		try:
			print(iter_n)
			trues = sess.run([top_k_op])
			true_count += np.sum(trues)
			iter_n += 1
		except tf.errors.OutOfRangeError:
			break

	accuracy = true_count / total_examples
	print(accuracy)


def eval_measure_2(sess, logits, labels):

	iter_n = 0
	predictions_all = np.array([])
	labels_all = np.array([])
	while True:
		try:
			print(iter_n)
			predictions, labels_r = sess.run([tf.argmax(logits, axis=1), labels])

			predictions_all = np.append(predictions_all, predictions)
			labels_all = np.append(labels_all, labels_r)

			iter_n += 1
		except tf.errors.OutOfRangeError:
			break

	print(labels_all.shape, predictions_all.shape)
	conf_matrix = confusion_matrix(labels_all, predictions_all)
	print(conf_matrix)

	precision_macro = precision_score(labels_all, predictions_all, average='macro')
	precision_weighted = precision_score(labels_all, predictions_all, average='weighted')
	print(precision_macro, precision_weighted)

	recall_macro = recall_score(labels_all, predictions_all, average='macro')
	recall_weighted = recall_score(labels_all, predictions_all, average='weighted')
	print(recall_macro, recall_weighted)

	accuracy = accuracy_score(labels_all, predictions_all)
	print(accuracy)


def eval_once(saver, eval_measure, logits, labels, args, checkpoint_dir):

	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print('No checkpoint file found')
			return

		eval_measure(sess, logits, labels, *args)


def eval(args):
	with tf.Graph().as_default():

		images, labels = args['inputs'](train=False)

		# logits = args['inference'](images, args['n_classes'])
		logits = model.inference(images, args['n_classes'], args['relu'], 
								 args['lrn'], args['pool_type'], args['trainable'])

		# top_k_op = tf.nn.in_top_k(logits, labels, 1)

		saver = tf.train.Saver()

		eval_once(saver, args['eval_measure'], logits, labels, 
				  args['eval_measure_args'], args['checkpoint_dir'])


def mnist_eval(args):

	args['inputs'] = read_input.mnist_input
	args['n_classes'] = 10

	if args['eval_measure_id'] == 1:
		args['eval_measure'] = eval_measure_1
		args['eval_measure_args'] = (10000,) # examples per epoch
	else:
		args['eval_measure'] = eval_measure_2
		args['eval_measure_args'] = tuple()
	
	eval(args)


def caltech_101_eval(args):
	args['inputs'] = read_input.caltech_101_input
	args['n_classes'] = 102

	if args['eval_measure_id'] == 1:
		args['eval_measure'] = eval_measure_1
		args['eval_measure_args'] = (30*102,) # examples per epoch
	else:
		args['eval_measure'] = eval_measure_2
		args['eval_measure_args'] = tuple()

	eval(args)


if __name__ == '__main__':
	args = read_command()

	if args['dataset'] == 'mnist':
		mnist_eval(args)
	elif args['dataset'] == 'caltech_101':
		caltech_101_eval(args)