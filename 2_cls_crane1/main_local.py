from nsml import DATASET_PATH
import nsml

import numpy as np
import glob
import argparse

from nnet_model import model_unet
from data_gen_func import DataGenerator
import parameters as param
from keras.models import load_model

def bind_model(model):
	def save(dir_name):
		np.save(dir_name + '/params.npy', np.array([model.all_mean, model.all_std]))

	def load(dir_name):
		params = np.load(dir_name + '/params.npy')
		model.all_mean = params[0]
		model.all_std = params[1]

	def infer(data):
		return model.forward(data)

	# DONOTCHANGE: They are reserved for nsml
	nsml.bind(save=save, load=load, infer=infer)


def collect_test_outputs(_data_gen_test, _data_out, quick_test):
	# Collecting ground truth for test data
	nb_batch = 2 if quick_test else _data_gen_test.get_total_batches()

	batch_size = _data_out[0][0]
	y = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2], _data_out[0][3]))
	y_hat = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2], _data_out[0][3]))

	print("nb_batch in test: {}".format(nb_batch))
	cnt = 0
	for oracle_in, oracle_out in _data_gen_test.generate(is_eval=True):
		y[cnt * batch_size:(cnt + 1) * batch_size, :, :] = oracle_in
		y_hat[cnt * batch_size:(cnt + 1) * batch_size, :, :] = oracle_out
		cnt = cnt + 1
		if cnt == nb_batch:
			break
	return y, y_hat

def feat_proc(in1, feat_len):

	in1 = np.concatenate([in1, np.zeros((in1.shape[0], feat_len-in1.shape[1]))+1e-6],axis=-1)

	t_all, k = in1.shape
	t = param.timestep
	if t_all < t:
		in1 = in1 / np.max(in1)
	else:
		in1 =  in1[:int(t_all/t)*t,:]
		in1_bat = np.reshape(in1, (int(t_all/t),t,k))


		nb_batch = in1_bat.shape[0]
		for b in range(nb_batch):
			norm_scale = np.max(in1_bat[b,:,:])
			in1_bat[b,:,:] = in1_bat[b,:,:] / norm_scale

	in1 = np.reshape(in1_bat,(*in1_bat.shape,1))
	return in1

class GaussianModel():
	
	def __init__(self):
		self.all_mean = 0
		self.all_std = 1
		
	def train(self, train_file_list):
		data_list = [np.load(file).flatten() for file in train_file_list]
		all_data = np.concatenate(data_list)
		self.all_mean = all_data.mean()
		self.all_std = all_data.std()
	
	def forward(self, data):
		# data : numpy matrix of shape (T * F ) where T is time and F is frequency
		# F = 500, T is not fixed
		m = data.mean()
		s = data.std()
		
		p = self.kl_divergence(m, s, self.all_mean, self.all_std)
		if p > 1:
			p = 1
		return p
		
	def kl_divergence(self, m1, s1, m2, s2):
		return (m1 - m2)**2 + np.log(s2/s1) + (s1**2 + (m1 - m2)**2)/(2*(s2**2)) - 0.5
	

	

if __name__ == '__main__':
	
	args = argparse.ArgumentParser()

	# DONOTCHANGE: They are reserved for nsml
	args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
	args.add_argument('--iteration', type=str, default='0',
					  help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
	args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
	
	config = args.parse_args()
	
	
	# Bind model
	###model = GaussianModel()
	###bind_model(model)
	
	
	# DONOTCHANGE: They are reserved for nsml
	# Warning: Do not load data before the following code!
	if config.pause:
		nsml.paused(scope=locals())
	
	TRAIN = 1
	TEST = 1

	if TRAIN:
		# Load data
		train_dataset_path = '../sample_data/2_cls_crane_a_samples'

		train_gen_val = DataGenerator(train_dataset_path, batch_size=param.batch_size,seq_len=param.timestep, feat_len=param.feat_len,
											n_channels=1, shuffle=True, per_file=False, is_eval=False)
		valid_gen_val = DataGenerator(train_dataset_path, batch_size=param.batch_size,seq_len=param.timestep, feat_len=param.feat_len,
											n_channels=1, shuffle=False, per_file=False, is_eval=True)

		# Load data generator for testing
		_, data_out = train_gen_val.get_data_sizes()
		y_oracle, y_hat = collect_test_outputs(valid_gen_val, data_out, param.quick_test)

		# Load model
		[model, encoder, decoder] = model_unet((param.timestep, param.feat_len, 1), param.batch_size)

		best_mae_metric = 99999
		best_epoch = -1
		patience_cnt = 0
		tr_loss = np.zeros(param.nb_epochs)
		val_loss = np.zeros(param.nb_epochs)
		mae_metric = np.zeros((param.nb_epochs, 2))
		nb_epoch = 2 if param.quick_test else param.nb_epochs

		# start training
		for epoch_cnt in range(nb_epoch):
		
			# train once per epoch
			hist = model.fit_generator(
				generator=train_gen_val.generate(is_eval=False),
				steps_per_epoch=2 if param.quick_test else train_gen_val.get_total_batches(),
				validation_data=valid_gen_val.generate(is_eval=True),
				validation_steps=2 if param.quick_test else valid_gen_val.get_total_batches(),
				epochs=param.epochs_per_fit,
				verbose=1
			)
			tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
			val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]
		
			# predict once per epoch
			out_val = model.predict_generator(
				generator=valid_gen_val.generate(is_eval=True),
				steps=2 if param.quick_test else valid_gen_val.get_total_batches(),
				verbose=1
			)
			y_h = out_val

			# Calculate the metrics
			mae_metric[epoch_cnt, 0] = np.mean(np.abs(y_h - y_oracle))
			print('===Validation metric(y) : {} ==='.format(np.mean(np.abs(y_h - y_oracle))))

			#Visualize intermediate results in spectrogram
			#wavPlot.spectrogram_batch(y_oracle,'figs/{}y'.format(epoch_cnt))
			#wavPlot.spectrogram_batch(y_h,'figs/{}x_h'.format(epoch_cnt))
			print('==={}-th epoch figure was saved==='.format(epoch_cnt))

			## Visualize the metrics with respect to param.epochs_per_fit
			#plot_functions(unique_name, tr_loss, val_loss, sed_metric, doa_metric, seld_metric)
		
			patience_cnt += 1
			if np.mean(mae_metric[epoch_cnt,:],axis=-1) < best_mae_metric:
				best_mae_metric = np.mean(mae_metric[epoch_cnt,:],axis=-1)
				best_epoch = epoch_cnt
				model.save(param.PATH_MODEL_CHECKPOINT)   
				patience_cnt = 0
		
			print(
					'best_epoch : %d\n' %
				(
					best_epoch
				)
			)

			if patience_cnt == param.patient:
				print('patient level reached. finishing training...')
				break
	
	if TEST:
		model = load_model('model/model.h5')
		
		#train_dataset_path = DATASET_PATH + '/train/train_data'
		train_dataset_path = '../sample_data/2_cls_crane_a_samples'
		train_data_files = sorted(glob.glob(train_dataset_path + '/*.npy')) 
		for file in train_data_files[:10]:
			y_in = feat_proc(np.load(file), param.feat_len)
			y_h = model.predict(y_in)
			dist = np.mean(np.abs(y_h - y_in))
			print(dist)


	### Save
	##epoch = 1
	##nsml.save(epoch) # If you are using neural networks, you may want to use epoch as checkpoints

	### Load test (Check if load method works well)
	##nsml.load(epoch)
	
	### Infer test
	##for file in train_data_files[:10]:
	##	data = np.load(file)
	##	print(model.forward(data))


