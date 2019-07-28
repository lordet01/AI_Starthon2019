import numpy as np
import keras
import random
from collections import deque
import os

class DataGenerator(object):
	'Generates data for Keras'
	def __init__(self, data_path_I='',
				 batch_size=16, seq_len=32, feat_len=512, n_channels=1, shuffle=True, per_file=False, is_eval=False):
		'Initialization'
		self._per_file = per_file
		self._seq_len = seq_len
		self._feat_len = feat_len
		self._dim = (self._seq_len, self._feat_len)
		self._batch_size = batch_size
		self._batch_seq_len = self._batch_size*self._seq_len
		self._circ_buf_feat_i = deque()
		self._circ_buf_feat_o1 = deque()
		self._circ_buf_feat_o2 = deque()
		self._filenames_list = self.__get_file_list__(data_path_I)
		self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
		self._n_channels = n_channels
		self._data_path_I = data_path_I
		self._shuffle = shuffle
		self._is_eval = is_eval
		self._nb_total_frames_ = self.__get_total_frames__()
		self.__on_epoch_end__()
		if self._per_file:
			self._nb_total_batches = len(self._filenames_list)
		else:
			self._nb_total_batches = int(np.floor((self._nb_total_frames_ / float(self._seq_len))))
		

	def __get_file_list__(self,path):
		import os
		'get file list from input folder'
		f_list = os.listdir(path)
		return f_list

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self._filenames_list) / self._batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self._batch_size:(index+1)*self._batch_size]

		# Find list of IDs
		_filenames_list_temp = [self._filenames_list[k] for k in indexes]

		# Generate data
		Y, Y_hat = self.generate(self._is_eval)

		return Y, Y_hat

	def __on_epoch_end__(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self._filenames_list))
		if self._shuffle == True:
			np.random.shuffle(self.indexes)

	def __get_total_frames__(self):
		file_cnt = 0
		self._nb_total_frames_ = 0
		for file_cnt in range(len(self._filenames_list)):
			try:
				temp_feat_i = np.load(os.path.join(self._data_path_I, self._filenames_list[file_cnt]))
			except:
				print('erronous feature file checked')
				print(os.path.join(self._data_path_I, self._filenames_list[file_cnt]))
			self._nb_total_frames_ += temp_feat_i.shape[0]
		return self._nb_total_frames_

	def __feat_norm__(self, data):
		'Batch-wise normalization'
		nb_batch = data.shape[0]
		for b in range(nb_batch):
			data[b,] /= np.max(data[b,]) 
		return data

	def __feat_norm_group__(self, in1):
		'Batch-wise normalization by input denorm scale'
		nb_batch = in1.shape[0]
		for b in range(nb_batch):
			norm_scale = np.max(in1[b,:,:,:])
			in1[b,:,:,:] = in1[b,:,:,:] / norm_scale
		return in1

	def __split_in_seqs__(self, data):
		if len(data.shape) == 1:
			if data.shape[0] % self._seq_len:
				data = data[:-(data.shape[0] % self._seq_len), :]
			data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, 1))
		elif len(data.shape) == 2:
			if data.shape[0] % self._seq_len:
				data = data[:-(data.shape[0] % self._seq_len), :]
			data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1]))
		elif len(data.shape) == 3:
			if data.shape[0] % self._seq_len:
				data = data[:-(data.shape[0] % self._seq_len), :, :]
			data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1], data.shape[2]))
		else:
			print('ERROR: Unknown data dimensions: {}'.format(data.shape))
			exit()
		return data

	def get_data_sizes(self):
		feat_shape = (self._batch_size, self._seq_len, self._feat_len, self._n_channels)
		return feat_shape, [feat_shape, feat_shape]

	def get_total_batches(self):
		return self._nb_total_batches

	def generate(self, is_eval):
		'Generates data containing _batch_size samples' # X : (n_samples, *_dim, _n_channels)

		while 1:
		#if 1:
			if self._shuffle and is_eval == False:
				random.shuffle(self._filenames_list)
	
			# Ideally this should have been outside the while loop. But while generating the test data we want the data
			# to be the same exactly for all epoch's hence we keep it here.
			self._circ_buf_feat_i = deque()
	
			file_cnt = 0
			##for i in range(len(self._filenames_list)):
			# load feat and label to circular buffer. Always maintain at least one batch worth feat and label in the
			# circular buffer. If not keep refilling it.
			buff_cnt = 0
			#print(buff_cnt)
			while buff_cnt < self._batch_seq_len:
				temp_feat_i = np.load(os.path.join(self._data_path_I, self._filenames_list[file_cnt]))
				temp_feat_i = np.concatenate([temp_feat_i, np.zeros((temp_feat_i.shape[0], self._feat_len-temp_feat_i.shape[1]))+1e-6],axis=-1)
				if np.sum(np.isnan(temp_feat_i)):
					print('!!!!!NaN Detected!!!!!!')
				else:
					for row_cnt, row in enumerate(temp_feat_i):
						self._circ_buf_feat_i.append(row)
						buff_cnt += 1
						
					# If self._per_file is True, this returns the sequences belonging to a single audio recording
					if self._per_file:
						extra_frames_i = self._batch_seq_len - temp_feat_i.shape[0]
						extra_feat_i = np.ones((extra_frames_i, temp_feat_i.shape[1])) * 1e-6

						for row_cnt, row in enumerate(extra_feat_i):
							self._circ_buf_feat_i.append(row)
							buff_cnt += 1

				file_cnt = file_cnt + 1

				#Reshuffle if the file is insufficient to make one batch
				if len(self._filenames_list) == file_cnt:
					file_cnt = 0
			#print(buff_cnt)

			# Read one batch size from the circular buffer
			feat_i = np.ones((self._batch_seq_len, self._feat_len * self._n_channels)) * 1e-6

			try:
				for j in range(self._batch_seq_len):
					feat_i[j, :] = self._circ_buf_feat_i.popleft()
			except:
				print('Buffer Error Detected')

			feat_i = np.reshape(feat_i, (self._batch_seq_len, self._feat_len, self._n_channels))

			# Split to sequences
			feat_i = self.__split_in_seqs__(feat_i)

			# Data normalization (Max norm per batch)
			feat_i = self.__feat_norm_group__(feat_i)
			
			# Output Feature
			feat_o = feat_i
			
			yield feat_i, feat_o
