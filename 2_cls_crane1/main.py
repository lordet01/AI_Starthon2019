from nsml import DATASET_PATH
import nsml

import numpy as np
import glob
import argparse
import os
from nnet_model import model_unet
from data_gen_func import DataGenerator
import parameters as param
##import wavPlot
from keras.models import load_model
from data_process import feat_proc, collect_test_outputs

def bind_model(model):
	def save(path):
		model.save(os.path.join(path, 'model.h5')) 
		print('----model saved----')

	def load(path):
		model = load_model(os.path.join(path, 'model.h5'))
		print('----model loaded----')
		return model

	def infer(data):
		return inference(model, data, config)

	# DONOTCHANGE: They are reserved for nsml
	nsml.bind(save=save, load=load, infer=infer)


def inference(model, data, config):
	y_in = feat_proc(data, param.feat_len)
	y_h = model.predict(y_in)
	dist = np.mean(np.abs(y_h - y_in))
	print(dist)
	return dist

	
##class GaussianModel():
##	
##	def __init__(self):
##		self.all_mean = 0
##		self.all_std = 1
##		
##	def train(self, train_file_list):
##		data_list = [np.load(file).flatten() for file in train_file_list]
##		all_data = np.concatenate(data_list)
##		self.all_mean = all_data.mean()
##		self.all_std = all_data.std()
##	
##	def forward(self, data):
##		# data : numpy matrix of shape (T * F ) where T is time and F is frequency
##		# F = 500, T is not fixed
##		m = data.mean()
##		s = data.std()
##		
##		p = self.kl_divergence(m, s, self.all_mean, self.all_std)
##		if p > 1:
##			p = 1
##		return p
##		
##	def kl_divergence(self, m1, s1, m2, s2):
##		return (m1 - m2)**2 + np.log(s2/s1) + (s1**2 + (m1 - m2)**2)/(2*(s2**2)) - 0.5
	

if __name__ == '__main__':
	
	args = argparse.ArgumentParser()

	# DONOTCHANGE: They are reserved for nsml
	args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
	args.add_argument('--iteration', type=str, default='0',
					  help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
	args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
	
	config = args.parse_args()
	
	
	# Bind model
	[model, encoder, decoder] = model_unet((param.timestep, param.feat_len, 1), param.batch_size)
	bind_model(model)
	
	
	# DONOTCHANGE: They are reserved for nsml
	# Warning: Do not load data before the following code!
	if config.pause:
		nsml.paused(scope=locals())
	

	if config.mode == "train":
		# Load data
		train_dataset_path = DATASET_PATH + '/train/train_data'

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

			patience_cnt += 1
			if np.mean(mae_metric[epoch_cnt,:],axis=-1) < best_mae_metric:
				best_mae_metric = np.mean(mae_metric[epoch_cnt,:],axis=-1)
				best_epoch = epoch_cnt
				nsml.save(1) # If you are using neural networks, you may want to use epoch as checkpoints
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
	
			
	### Load test (Check if load method works well)
	##nsml.load(epoch)
	
	### Infer test
	##for file in train_data_files[:10]:
	##	data = np.load(file)
	##	print(model.forward(data))


