"""
This file combines methods from files model.py, read_input.py to perform
training on the dataset.
"""

from read_args import read_command
from datetime import datetime
import time

import tensorflow as tf
import model
import read_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('log_frequency', 10, 'frequency to show logs')
tf.app.flags.DEFINE_integer('max_steps', 2000, 'maximum number of steps to train')
tf.app.flags.DEFINE_string('c', '', 'directory to store checkpoints')
tf.app.flags.DEFINE_boolean('no-train', True, 'whether to train the inside weights')

def train(args):

	with tf.Graph().as_default():

		global_step = tf.train.get_or_create_global_step()

		images, labels = args['inputs'](True)

		# logits = args['inference'](images, args['n_classes'])
		logits = model.inference(images, args['n_classes'], args['relu'], 
								 args['lrn'], args['pool_type'], args['trainable'])

		loss = model.loss(logits, labels)

		train_op = model.train(loss, global_step, args['examples_per_epoch'])

		class _LoggerHook(tf.train.SessionRunHook):

			def begin(self):
				self._step = -1
				self._start_time = time.time()

			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss)

			def after_run(self, run_context, run_values):
				if self._step % FLAGS.log_frequency == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time

					loss_value = run_values.results
					examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
					sec_per_batch = duration / FLAGS.log_frequency

					format_str = ('%s: step %d, loss = %.2f (%.2f examples per second, '
						'%.2f seconds per batch)')

					print(format_str % (datetime.now(), self._step, loss_value, 
										examples_per_sec, sec_per_batch))

		with tf.train.MonitoredTrainingSession(
			checkpoint_dir=args['checkpoint_dir'],
			hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
				   tf.train.NanTensorHook(loss),
				   _LoggerHook()]) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(train_op)


def train_2(args):

	with tf.Graph().as_default():

		global_step = tf.train.get_or_create_global_step()

		cond = lambda i, _: tf.less(i, FLAGS.max_steps)

		def body(iter_n, start_time):

			images, labels = args['inputs'](True)

			logits = args['inference'](images, args['n_classes'])

			loss = model.loss(logits, labels)

			train_op = model.train(loss, global_step, args['examples_per_epoch'])

			def true_fn():
				# current_time = tf.constant(time.time())
				# duration = tf.subtract(current_time, start_time)
				# start_time = current_time

				# cur_time = tf.constant(datetime.now())
				# examples_per_sec = tf.constant(FLAGS.log_frequency * FLAGS.batch_size // duration)
				# sec_per_batch = tf.constant(duration // FLAGS.log_frequency)
				# pr = tf.print(tf.no_op(), [cur_time, iter_n, loss, examples_per_sec, sec_per_batch], 
				# 		 message='time, step, loss, examples per second, seconds per batch: ')
				pr = tf.Print(loss, [iter_n, loss], message='step, loss: ')
				return pr

			cond_op = tf.cond(tf.equal(tf.mod(iter_n, FLAGS.log_frequency), 0), true_fn=true_fn,
							  false_fn=lambda : tf.constant(0.0))

			with tf.control_dependencies([cond_op, train_op]):
				return tf.add(iter_n, 1.0), start_time

		while_loop = tf.while_loop(cond, body, [tf.constant(0.0), tf.constant(time.time())], 
			 					   parallel_iterations=1)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(while_loop)


def mnist_train(args):

	args['inputs'] = read_input.mnist_input
	args['n_classes'] = 10
	args['examples_per_epoch'] = 60000
	train(args)

def caltech_101_train(args):

	args['inputs'] = read_input.caltech_101_input
	args['n_classes'] = 102
	args['examples_per_epoch'] = 102 * 30
	train(args)

if __name__ == '__main__':
	args = read_command()

	if args['dataset'] == 'mnist':
		mnist_train(args)
	elif args['dataset'] == 'caltech_101':
		caltech_101_train(args)