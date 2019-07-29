from __future__ import absolute_import, division, print_function

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Reshape, LSTM, GRU, Dropout, Add, multiply, add, dot, TimeDistributed, subtract
from keras.layers import Conv2D, BatchNormalization, ReLU, LeakyReLU, Conv2DTranspose, concatenate, Concatenate, Bidirectional, MaxPooling2D
from keras.layers import Activation, ConvLSTM2D, Permute, Flatten
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras import losses
from keras.utils import plot_model
from keras.regularizers import l1, l2

import numpy as np

##import matplotlib.pyplot as plt

print(tf.__version__)

def model_unet(input_shape, BATCH_SIZE):

	t,k,c = input_shape
	
	#----U-net Encoder----
	y = Input(shape=(t,k,1))#shape=input_shape ) #(B,t=128,f=k, c=1)
	conv1 = conv_bat_relu(y,16, (5,5), 2) #(1,64,256,16)
	conv2 = conv_bat_relu(conv1,32, (5,5), 2) #(1,32,128,32)
	conv3 = conv_bat_relu(conv2,64, (5,5), 2) #(1,16,64,64)
	conv4 = conv_bat_relu(conv3,128, (5,5), 2) #(1,8,32,128)
	conv5 = conv_bat_relu(conv4,256, (5,5), 2) #(1,4,16,256)
	conv6 = conv_bat_relu(conv5,k, (5,5), 2) #(1,2,8,k)
	
	Unet_encoder = Model(y, [conv6, conv5, conv4, conv3, conv2, conv1], name = 'Enc_codes')  
	Unet_encoder.summary()
	#plot_model(Unet_encoder, to_file='model/Unet_encoder.png', show_shapes=True)
	#----U-net Decoder----
	d_y = Input(shape=( t,k,1)) #(B,t=8,f=k, c=1)
	
	d_conv6 = Input(shape=( 2,8,k))
	d_conv5 = Input(shape=( 4,16,256))
	d_conv4 = Input(shape=( 8,32,128)) 
	d_conv3 = Input(shape=( 16,64,64))
	d_conv2 = Input(shape=( 32,128,32))
	d_conv1 = Input(shape=( 64,256,16))
	conc0 = d_conv6
	deconv1 = deconv_bat_relu(conc0,256, (5,5), 2)
	conc1 = concatenate([deconv1,d_conv5])
	deconv2 = deconv_bat_relu(conc1, 128, (5,5), 2)
	conc2 = concatenate([deconv2,d_conv4])
	deconv3 = deconv_bat_relu(conc2, 64, (5,5), 2)
	conc3 = concatenate([deconv3,d_conv3])
	deconv4 = deconv_bat_relu(conc3, 32, (5,5), 2)
	conc4 = concatenate([deconv4,d_conv2])
	deconv5 = deconv_bat_relu(conc4, 16, (5,5), 2)
	conc5 = concatenate([deconv5, d_conv1])
	irm_y = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid')(conc5)
	irm_y = Lambda(lambda x:x[:,:,:k])(irm_y)
	y_hat = multiply([irm_y, d_y])

	#U-net Decoder
	Unet_decoder = Model([d_y, \
							d_conv6, d_conv5, d_conv4, d_conv3, d_conv2, d_conv1], y_hat, name = 'Dec_outs')
	Unet_decoder.summary()
	#plot_model(TAUnet_decoder, to_file='model/Unet_decoder.png', show_shapes=True)
	#U-net Model
	enc_outs = Unet_encoder(y)
	dec_out = Unet_decoder([y, *enc_outs])
	y_hat_m = Reshape((t,k,1), name='Out_y_hat')(dec_out)
	
	Unet = Model(y, y_hat_m, name='Unet')
	Unet.summary()
 
	Unet.compile( loss= {'Out_y_hat' : 'mean_absolute_error'},
					loss_weights = {'Out_y_hat' : 1.0},
					optimizer='adam',
					#optimizer=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
					metrics=['mean_absolute_error'] )
	Unet.summary()
	model = Unet
	encoder = Unet_encoder
	decoder = Unet_decoder


	return model, encoder, decoder


def model_cnn_vae(input_shape, BATCH_SIZE):

	t,k,c = input_shape

	# Parameters
	latent_dim = 10
	epsilon_std = 0.1

	#----U-net Encoder----
	y = Input(shape=(t,k,1))#shape=input_shape ) #(B,t=128,f=k, c=1)
	conv1 = conv_bat_relu(y,16, (5,5), 2) #(1,64,256,16)
	conv2 = conv_bat_relu(conv1,32, (5,5), 2) #(1,32,128,32)
	conv3 = conv_bat_relu(conv2,64, (5,5), 2) #(1,16,64,64)
	conv4 = conv_bat_relu(conv3,128, (5,5), 2) #(1,8,32,128)
	conv5 = conv_bat_relu(conv4,128, (5,5), 2) #(1,4,16,256)
	conv6 = conv_bat_relu(conv5,128, (5,5), 2) #(1,2,8,512)
	
	#----VAE encoder----
	vae_enc_in = Flatten()(conv6)
	e_dense1 = Dense(1024, activation='relu')(vae_enc_in)  
	e_dense2 = Dense(512, activation='relu')(e_dense1)  
	e_dense3 = Dense(256, activation='relu')(e_dense2)  

	z_mean = Dense(latent_dim)(e_dense3)  
	z_log_var = Dense(latent_dim, activation=tf.nn.softplus)(e_dense3)

	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(1, latent_dim),
								mean=0., stddev=epsilon_std)
		return z_mean + K.exp(z_log_var) * epsilon

	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

	encoder = Model(y, [z_mean, z_log_var, z], name = 'Enc_codes')  
	encoder.summary()
	#plot_model(Unet_encoder, to_file='model/Unet_encoder.png', show_shapes=True)

	
	#----U-net Decoder----
	d_z = Input(shape=(latent_dim,), name='z_sampling')
	


	d_dense5 = Dense(256, activation=tf.nn.relu)(d_z)
	d_dense4 = Dense(512, activation=tf.nn.relu)(d_dense5)  
	d_dense3 = Dense(1024, activation=tf.nn.relu)(d_dense4)  
	d_dense2 = Dense(2048, activation=tf.nn.relu)(d_dense3)  
	vae_dec_out = Reshape((2,8,128))(d_dense2) 

	conc0 = vae_dec_out
	deconv1 = deconv_bat_relu(conc0,128, (5,5), 2)
	conc1 = deconv1
	deconv2 = deconv_bat_relu(conc1, 128, (5,5), 2)
	conc2 = deconv2
	deconv3 = deconv_bat_relu(conc2, 64, (5,5), 2)
	conc3 = deconv3
	deconv4 = deconv_bat_relu(conc3, 32, (5,5), 2)
	conc4 = deconv4
	deconv5 = deconv_bat_relu(conc4, 16, (5,5), 2)
	conc5 = deconv5
	y_hat = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid')(conc5)
	y_hat = Lambda(lambda x:x[:,:,:k])(y_hat)

	#U-net Decoder
	decoder = Model(d_z, y_hat, name = 'Dec_outs')
	decoder.summary()
	#plot_model(TAUnet_decoder, to_file='model/Unet_decoder.png', show_shapes=True)
	#U-net Model
	enc_outs = encoder(y)
	dec_out = decoder(enc_outs[2])
	y_hat_m = Reshape((t,k,1), name='Out_y_hat')(dec_out)
	
	model = Model(y, y_hat_m, name='Unet_vae')
 
	# Define loss
	def vae_loss(x, x_hat):
	    # NOTE: binary_crossentropy expects a batch_size by dim for x and x_hat, so we MUST flatten these!
	    x = K.flatten(x)
	    x_hat = K.flatten(x_hat)
	    xent_loss = losses.mean_squared_error(x, x_hat)
	    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	    return xent_loss +  kl_loss

	model.compile(  loss = vae_loss,
					loss_weights = {'Out_y_hat' : 1.0},
					optimizer='adam',
					#optimizer=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
					metrics=['mean_absolute_error'] )
	model.summary()


	return model, encoder, decoder


def model_dnn(input_shape, BATCH_SIZE):

	t,k,c = input_shape
	
	#----U-net Encoder----
	y = Input(shape=(t,k,1))#shape=input_shape ) #(B,t=128,f=k, c=1)
	y_flat = Flatten()(y)
	hidden = Dense(64,activation='relu')(y_flat)
	y_hat = Dense(t*k,activation='relu')(hidden)
	y_hat_m = Reshape((t,k,1), name='Out_y_hat')(y_hat)
	
	model = Model(y, y_hat_m, name='DNN')
	model.summary()
 
	model.compile( loss= {'Out_y_hat' : 'mean_absolute_error'},
					loss_weights = {'Out_y_hat' : 1.0},
					optimizer='adam',
					#optimizer=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
					metrics=['mean_absolute_error'] )
	model.summary()


	return model


# Define loss
def unet_loss(x, x_hat):
	mae_loss = K.mean(K.abs(x - x_hat), axis=-1)
	return mae_loss

# spectrogram based Unet architecture
def conv_bat_relu(X,filters, kernel_size, strides):
	out = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer='he_normal', padding='same')(X)
	out = BatchNormalization(axis=-1)(out)
	out = LeakyReLU(0.2)(out)
	out = MaxPooling2D(pool_size=strides)(out)
	out = Dropout(0.2)(out)
	return out

# spectrogram based Unet architecture
def deconv_bat_relu(X,filters, kernel_size, strides):
	out = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(X)
	out = BatchNormalization(axis=-1)(out)
	out = ReLU()(out)
	out = Dropout(0.2)(out)
	return out
