"""
This file is used to read the command line arguments
"""
import argparse
import model

def get_inference(inference):

	if inference == 'inference_1_1':
		return model.inference_1_1
	elif inference == 'inference_1_2':
		return model.inference_1_2
	elif inference == 'inference_2_1':
		return model.inference_2_1
	elif inference == 'inference_3_1':
		return model.inference_3_1
	elif inference == 'inference_4_1':
		return model.inference_4_1
	else:
		raise Exception('Invalid inference')


def read_command():

	parser = argparse.ArgumentParser()

	parser.add_argument('dataset', metavar='Dataset', help='dataset to perform training', 
						type=str)

	# parser.add_argument('inference', metavar='Inference', help='method of performing inference',
	# 					type=str)

	parser.add_argument('-c', dest='checkpoint', metavar='Checkpoint', 
						help='directory to store checkpoints '
						'(dafault value is inferred from infernece)', type=str, default='')

	parser.add_argument('-max_steps', dest='max_steps', metavar='MAXsteps', 
						help='max number of steps to train '
						'(dafault value is 1000)', type=int, default=1000)

	parser.add_argument('-batch_size', dest='batch_size', metavar='BatchSize', 
						help='batch size for dataset '
						'(dafault value is 128)', type=int, default=128)

	parser.add_argument('-e', dest='eval_measure', metavar='EvalMeasure', 
						help='Eval measure to use '
						'(dafault value is 2)', type=int, default=2)

	parser.add_argument('--relu', dest='relu', metavar='Relu', action='store_const',
						const=True, default=False, help='Relu to use or not '
						'(dafault value is false)')	

	parser.add_argument('--lrn', dest='lrn', metavar='Lrn', action='store_const',
						const=True, default=False, help='Lrn to use or not '
						'(dafault value is false)')	

	parser.add_argument('-p', dest='pool_type', metavar='PoolingType', 
						help='Pooling type to use '
						'(dafault value is avg)', type=str, default='avg')

	parser.add_argument('--no-train', dest='trainable', metavar='Trainable', 
						help='Whether inside weights are trainable '
						'(dafault value is True)', const=False, default=True, action='store_const')

	cmds = parser.parse_args()

	args = {}
	args['dataset'] = cmds.dataset
	# args['inference'] = get_inference(cmds.inference)

	if cmds.checkpoint == '':
		# args['checkpoint_dir'] = './checkpoints/' + cmds.dataset + '/' + cmds.inference
		args['checkpoint_dir'] = './checkpoints/' + cmds.dataset + '/inference'

		if cmds.relu:
			args['checkpoint_dir'] += '_relu'

		if cmds.lrn:
			args['checkpoint_dir'] += '_lrn'

		if cmds.pool_type == 'avg':
			args['checkpoint_dir'] += '_avg'
		else:
			args['checkpoint_dir'] += '_max'			

		if not cmds.trainable:
			args['checkpoint_dir'] += '_False'

		args['checkpoint_dir'] += '/'

	else:
		args['checkpoint_dir'] = cmds.checkpoint

	args['relu'] = cmds.relu
	args['lrn'] = cmds.lrn
	args['pool_type'] = cmds.pool_type
	args['trainable'] = cmds.trainable

	args['eval_measure_id'] = cmds.eval_measure

	print(args)
	return args
