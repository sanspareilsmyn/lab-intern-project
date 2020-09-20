import numpy as np
import scipy.io
from numpy import linalg as LA
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math

N=64 # the number of users
M=32 # spreading factor
L=1 # frame length

''' import dataset generated from data_generation.py '''
filename='dataset_32_20_32_1.npy'
data = np.load(filename).item()
print('Training dataset:\n-------------------')
print('x:', data['train']['x'].shape)
print('y:', data['train']['y'].shape)

print('Validation dataset:\n-------------------')
print('x:', data['valid']['x'].shape)
print('y:', data['valid']['y'].shape)

print('Test dataset:\n-------------------')
print('x:', data['test']['x'].shape)
print('y:', data['test']['y'].shape)

##########################################################################
class AE(object):
	def __init__(self, filename=None):  
		self.M = M
		self.N = N
		self.L = L
		self.graph = None
		self.sess= None
		self.vars = None
		self.saver = None
		self.create_graph()
		self.create_session()
		if filename is not None:
			self.load(filename)
		return
	def create_graph(self):
		'''the computation graph of the autoencoder'''
		self.graph = tf.Graph()
		with self.graph.as_default():
			batch_size = tf.placeholder(tf.int64, shape=())
			BN_phase = tf.placeholder(tf.bool)

			# Channel
			noise_std = tf.placeholder(tf.float32, shape=())
			AWGN_real = tf.random_normal(shape=[batch_size, self.L, self.M], mean=0.0, stddev=noise_std)

			# Transmitter
			x = tf.placeholder(dtype=tf.float32, shape=[None, self.L, self.N])   # BPSK signaling from N users
			x_label = tf.placeholder(dtype=tf.float32, shape=[None, self.N])   # label for activity

			y = self.spread_matrix(x) + AWGN_real # [batch_size, L, M]

			# Receiver
			prediction = self.AUD_Prediction_Layer(y, BN_phase)
			prediction_l = tf.nn.sigmoid(prediction)

			# Loss function
			cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x_label, logits=prediction), axis=1))

			# metric: accuracy rate
			sparsity = tf.cast(tf.count_nonzero(x_label, axis=1), tf.float32) # for bernoulli sampling
			
			# prediction: when basestation does not know sparsity
			multi_hot_prediction = tf.cast(tf.greater(prediction_l, 4e-1), dtype=tf.float32)

			how_many_correct = tf.reduce_sum(tf.multiply(multi_hot_prediction, x_label), axis=1)
			union_of_predict_target = tf.cast(tf.count_nonzero(tf.add(multi_hot_prediction, x_label), axis=1), tf.float32)
			accuracy = tf.reduce_mean(tf.divide(how_many_correct, union_of_predict_target))

			# Optimizer
			lr = tf.placeholder(tf.float32, shape=()) # we can feed in any desired learning rate for each step
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

			# References to graph variables we need to access later
			self.vars = {
				'x' : x,
				'x_label' : x_label,
				'accuracy': accuracy,
				'batch_size': batch_size,
				'cross_entropy': cross_entropy,
				'init': tf.global_variables_initializer(),
				'lr': lr,
				'noise_std': noise_std,
				'train_op': train_op,
				'BN_phase': BN_phase,
				'prediction': prediction_l
			}
			self.saver = tf.train.Saver()
		return

	def create_session(self):
		'''Create a session for the autoencoder instance with the computational graph'''
		self.sess = tf.Session(graph=self.graph)
		self.sess.run(self.vars['init'])
		return

	def spread_matrix(self, input):
		W1 = tf.get_variable("W_real", shape=[1, self.N, self.M], initializer=tf.truncated_normal_initializer(stddev=1.0), trainable=True)     # filter dimension: [1(filter_width), N(in_channels), M(out_channels)] 
		Normalized_W1 = tf.div(W1, tf.reshape(tf.norm(W1, axis=2), shape=[1, self.N, 1]), name='Normalized_W1')
		Shx = tf.nn.conv1d(input, Normalized_W1, stride=1, padding='SAME')  # input dimension: [batch, 2*L(in_width), N(in_channels))]
		return Shx

	def AUD_Prediction_Layer(self, input, BN_phase):
		# Initialization
		y = tf.reshape(input, shape=[-1, self.L * self.M]) # flatten the input
		y = tf.layers.dense(y, 5*self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y = tf.layers.batch_normalization(y, training=BN_phase)
		y = tf.nn.relu(y)

		# 5 Times Repeat
		y2 = tf.layers.dense(y, 5*self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y2 = tf.layers.batch_normalization(y2, training=BN_phase)
		y2 = tf.nn.relu(y2)
		y2 = tf.layers.dense(y2, 5*self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y2 = tf.layers.batch_normalization(y2, training=BN_phase)
		y3 = y2 + y

		y3 = tf.nn.relu(y3) # 1회차

		y4 = tf.layers.dense(y3, 5*self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y4 = tf.layers.batch_normalization(y4, training=BN_phase)
		y4 = tf.nn.relu(y4)
		y4 = tf.layers.dense(y4, 5*self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y4 = tf.layers.batch_normalization(y4, training=BN_phase)
		y5 = y4 + y3

		y5 = tf.nn.relu(y5) # 2회차

		y6 = tf.layers.dense(y5, 5 * self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y6 = tf.layers.batch_normalization(y6, training=BN_phase)
		y6 = tf.nn.relu(y6)
		y6 = tf.layers.dense(y6, 5 * self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y6 = tf.layers.batch_normalization(y6, training=BN_phase)
		y7 = y6 + y5

		y7 = tf.nn.relu(y7) # 3회차

		y8 = tf.layers.dense(y7, 5 * self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y8 = tf.layers.batch_normalization(y8, training=BN_phase)
		y8 = tf.nn.relu(y8)
		y8 = tf.layers.dense(y8, 5 * self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y8 = tf.layers.batch_normalization(y8, training=BN_phase)
		y9 = y8 + y7

		y9 = tf.nn.relu(y9) # 4회차

		y10 = tf.layers.dense(y9, 5 * self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y10 = tf.layers.batch_normalization(y10, training=BN_phase)
		y10 = tf.nn.relu(y10)
		y10 = tf.layers.dense(y10, 5 * self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y10 = tf.layers.batch_normalization(y10, training=BN_phase)
		y11 = y10 + y9

		y11 = tf.nn.relu(y11) # 5회차

		y12 = tf.layers.dense(y11, 5 * self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y12 = tf.layers.batch_normalization(y12, training=BN_phase)
		y12 = tf.nn.relu(y12)
		y12 = tf.layers.dense(y12, 5 * self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())
		y13 = y12 + y11

		y_output = tf.layers.dense(y13, self.N, kernel_initializer=tf.contrib.layers.xavier_initializer())

		return y_output

	def EbNo2Sigma(self, ebnodb):
		'''Convert Eb/No in dB to noise standard deviation'''
		ebno = 10**(ebnodb/10)
		return 1/np.sqrt(ebno)
	
	def gen_feed_dict(self, x, x_label, batch_size=None, ebnodb=None, lr=None, BN_phase=None):
		'''generate a feed dictionary for training and validation'''
		return {
		self.vars['x']: x,
		self.vars['x_label']: x_label,
		self.vars['batch_size']: batch_size,
		self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
		self.vars['lr']: lr,
		self.vars['BN_phase']: BN_phase
		}

	def load(self, filename_l):
		'''load an pre-trained model'''
		self.saver = tf.train.import_meta_graph(filename_l+'.meta')
		return self.saver.restore(self.sess, filename_l)
		#return self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))

	def save(self, filename_s):
		'''save the current model'''
		return self.saver.save(self.sess, filename_s)

	def train(self, training_params, validation_params):
		'''training and validation loop'''
		for index, params in enumerate(training_params):
			train_x, train_y, batch_size, lr, ebnodb, epochs = params
			print('\nBatch Size: ' + str(batch_size) +
				', Learning Rate: ' + str(lr) +
				', EbNodB: ' + str(ebnodb) +
				', epochs: ' + str(epochs))
			valid_x, valid_y, val_size, val_ebnodb, val_steps = validation_params[index]
			train_size = train_x.shape[0]
			indices = np.arange(train_size)

			for e in range(epochs):				
				for i in range(int(train_size / batch_size)):
					inds = indices[i*batch_size:(i+1)*batch_size]
					batch_x = train_x[inds]
					batch_y = train_y[inds]

					ebnodb_random = np.random.uniform(low=ebnodb, high=ebnodb+5, size=1)

					self.train_step(batch_x, batch_y, batch_size, ebnodb_random[0], lr)	
					if (i%val_steps==0):
						train_acc, train_loss = self.test_step(batch_x, batch_y, batch_size, ebnodb_random[0])
						print('[Train] step %d, epoch %d: %f, %f' %(i, e, train_acc, train_loss))
						val_acc, val_loss = self.test_step(valid_x, valid_y, val_size, val_ebnodb)
						print('[Valid] step %d, epoch %d: %f, %f' %(i, e, val_acc, val_loss))
				np.random.shuffle(indices) # shuffle dataset after each epoch training
		return

	def test(self, test_params):
		for index, params in enumerate(test_params):
			test_x, test_y, test_size, test_ebnodb = params
			print('\ntest Size: ' + str(test_size) + ', SNR: ' + str(test_ebnodb))
			test_acc, test_loss = self.test_step(test_x, test_y, test_size, test_ebnodb)
			print('[Test] %f, %f' %(test_acc, test_loss))
		return

	def train_step(self, batch_x, batch_y, batch_size, ebnodb, lr):
		'''A single training step'''
		self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(batch_x, batch_y, batch_size, ebnodb, lr, True))
		return

	def test_step(self, x, y, batch_size, ebnodb):
		'''compute the accuracy over a single batch and Eb/No'''
		accuracy, loss= self.sess.run([self.vars['accuracy'], self.vars['cross_entropy']], feed_dict=self.gen_feed_dict(x, y, batch_size, ebnodb, 0, False))
		return accuracy, loss

##########################################################################

train_EbNodB = 15
val_EbNodB = 15

training_params = [
    #batch_size, lr, ebnodb, iterations
    [data['train']['x'], data['train']['y'], 200, 0.01, 15, 10],
    [data['train']['x'], data['train']['y'], 200, 0.001, 15, 10],
    [data['train']['x'], data['train']['y'], 200, 0.0001, 15, 10],
    [data['train']['x'], data['train']['y'], 200, 0.00001, 15, 10]
]

validation_params = [
    #batch_size, ebnodb, val_steps 
    [data['valid']['x'], data['valid']['y'], 60000, val_EbNodB, 1000],
    [data['valid']['x'], data['valid']['y'], 60000, val_EbNodB, 1000],
    [data['valid']['x'], data['valid']['y'], 60000, val_EbNodB, 1000],
    [data['valid']['x'], data['valid']['y'], 60000, val_EbNodB, 1000]
]
# test_EbNodB = 15
test_params = [
	[data['test']['x'], data['test']['y'], 60000, 5],
	[data['test']['x'], data['test']['y'], 60000, 7],
	[data['test']['x'], data['test']['y'], 60000, 9],
	[data['test']['x'], data['test']['y'], 60000, 10],
	[data['test']['x'], data['test']['y'], 60000, 12],
	[data['test']['x'], data['test']['y'], 60000, 14],
	[data['test']['x'], data['test']['y'], 60000, 15],
	[data['test']['x'], data['test']['y'], 60000, 16],
	[data['test']['x'], data['test']['y'], 60000, 18],
	[data['test']['x'], data['test']['y'], 60000, 20],
	[data['test']['x'], data['test']['y'], 60000, 22],
	[data['test']['x'], data['test']['y'], 60000, 25]
]

tf.reset_default_graph()

ae = AE('Model6_SNR15to20_N64_M32_P1to25_Leanred')

op_to_restore = ae.graph.get_tensor_by_name('Normalized_W1:0')
S=ae.sess.run(op_to_restore)
scipy.io.savemat('Model6_1to25_S.mat', dict(Model6_1to25_S=S))
print(S[0,0])