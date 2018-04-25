import read_args
import train

if __name__ == '__main__':

	args = {}
	# args['dataset'] = 'caltech_101'

	# inference_list = ['inference_2_1', 'inference_3_1', 'inference_4_1']

	# for inference in inference_list:
	# 	args['inference'] = read_args.get_inference(inference)
	# 	args['checkpoint_dir'] = './checkpoints/' + 'caltech_101' + '/' + inference + '/'
	# 	train.caltech_101_train(args)

	args['dataset'] = 'caltech_101'
	args['pool_type'] = 'avg'
	args['eval_measure_id'] = 2

	checkpoint_parent = './checkpoints/caltech_101/inference'

	# args['checkpoint_dir'] = checkpoint_parent + '_lrn_avg'
	# args['lrn'] = True
	# args['relu'] = False
	# args['trainable'] = True
	# train.caltech_101_train(args)

	args['checkpoint_dir'] = checkpoint_parent + '_lrn_avg_False'
	args['lrn'] = True
	args['relu'] = False
	args['trainable'] = False
	train.caltech_101_train(args)

	args['checkpoint_dir'] = checkpoint_parent + '_relu_lrn_avg'
	args['lrn'] = True
	args['relu'] = True
	args['trainable'] = True
	train.caltech_101_train(args)
