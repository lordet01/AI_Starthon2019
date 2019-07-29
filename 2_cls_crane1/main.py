from nsml import DATASET_PATH
import nsml

import numpy as np
import glob
import argparse
import os
from nnet_model import model_cnn_vae
from data_gen_func import DataGenerator
import parameters as param
from keras.models import load_model
from data_process import feat_proc, collect_test_outputs

def bind_model(model_vae):
	def save(path):
		model_vae.save(os.path.join(path, 'model_vae.h5'))
		print('----model (vae) saved----')

	def load(path):
		model_vae = load_model(os.path.join(path, 'model_vae.h5'), compile=False)
		print('----model (vae) loaded----')
		return model_vae

	def infer(data):
		return inference(model_vae, data, config)

	# DONOTCHANGE: They are reserved for nsml
	nsml.bind(save=save, load=load, infer=infer)

def inference(model, data, config):
	##noise = np.abs(np.random.rand((*data.shape)))## * np.max(data) #make atypical scenarios
	##data = data + noise
	y_in = feat_proc(data, param.feat_len)
	y_hat = model.predict(y_in)
	norm_val = np.mean(np.abs(y_in[:,:,:500,:] + y_hat[:,:,:500,:]))
	dist = np.abs(y_in - y_hat)
	dist = np.mean(dist[:,:,:500,:]) / norm_val
	##enc_out = encoder.predict(y_in)
	##z_mean = enc_out[0]
	##z_logvar = enc_out[1]
	##mean_now = np.mean(z_mean,axis=0)
	##std_now = np.sqrt(np.mean(np.exp(z_logvar) + np.power(z_mean - mean_now,2),axis=0))
	##dist = model_Gaussian.forward(mean_now,std_now)
	print(dist)
	return dist

class GaussianModel():
	
	def __init__(self):
		self.all_mean = 0
		self.all_std = 1
		
	def train(self, mean_normal, var_normal):
		self.all_mean = mean_normal
		self.all_std = var_normal
	
	def forward(self, m, s):
		p = self.kl_divergence(m, s, self.all_mean, self.all_std)
		if p > 1.0:
			p = 1
		return p
		
	def kl_divergence(self, m1, s1, m2, s2):
		return np.mean((m1 - m2)**2 + np.log(s2/s1) + (s1**2 + (m1 - m2)**2)/(2*(s2**2)) - 0.5)

if __name__ == '__main__':
	
	args = argparse.ArgumentParser()

	# DONOTCHANGE: They are reserved for nsml
	args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
	args.add_argument('--iteration', type=str, default='0',
					  help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
	args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
	
	config = args.parse_args()
	
	
	# Bind model
	[model, encoder, decoder] = model_cnn_vae((param.timestep, param.feat_len, 1), param.batch_size)
	model_Gaussian = GaussianModel()
	
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

		# start training
		best_mae_metric = 99999
		best_epoch = -1
		patience_cnt = 0
		nb_epoch = 2 if param.quick_test else param.nb_epochs

		# train once per epoch
		for epoch_cnt in range(nb_epoch):
			hist = model.fit_generator(
				generator=train_gen_val.generate(is_eval=False),
				steps_per_epoch=2 if param.quick_test else train_gen_val.get_total_batches(),
				validation_data=valid_gen_val.generate(is_eval=True),
				validation_steps=2 if param.quick_test else valid_gen_val.get_total_batches(),
				epochs=param.epochs_per_fit,
				verbose=1
			)
			nsml.save(epoch_cnt) # If you are using neural networks, you may want to use epoch as checkpoints 

		####Get Normal GMM
		###enc_out = encoder.predict_generator(
		###	generator=valid_gen_val.generate(is_eval=True),
		###	steps=2 if param.quick_test else valid_gen_val.get_total_batches(),
		###	verbose=1)	
		###z_mean = enc_out[0]
		###z_logvar = enc_out[1]
		###mean_normal = np.mean(z_mean,axis=0)
		###std_normal = np.sqrt(np.mean(np.exp(z_logvar) + np.power(z_mean - mean_normal,2),axis=0))
		###model_Gaussian.train(mean_normal,std_normal)
					


