from __future__ import absolute_import, division, print_function

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Reshape, LSTM, GRU, Dropout, Add, multiply, add, dot, TimeDistributed, subtract
from keras.layers import Conv2D, BatchNormalization, ReLU, LeakyReLU, Conv2DTranspose, concatenate, Concatenate, Bidirectional, MaxPooling2D
from keras.layers import Activation, ConvLSTM2D, Permute
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
