import numpy as np
import parameters as param

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