#    Implements state-of-the-art deep learning architectures.
#
#    Copyright (C) 2017  Antonio Montieri
#    email: antonio.montieri@unina.it
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on 13 nov 2017

@author: antonio
'''

import sys, os, time, errno, datetime

import keras, sklearn, numpy, scipy, imblearn.metrics
import tensorflow as tf
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D,\
			 Conv2D, MaxPooling2D,\
			 Dense, LSTM, GRU, Flatten, Dropout, Reshape, BatchNormalization,\
			 Input, Bidirectional, concatenate
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.constraints import maxnorm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from yadlt.models.autoencoders.stacked_denoising_autoencoder import StackedDenoisingAutoencoder

import pickle
import logging
from logging.handlers import RotatingFileHandler
from pprint import pprint

tf.logging.set_verbosity(tf.logging.INFO)



class DLArchitecturesException(Exception):
	'''
	Exception for DLArchitectures class
	'''
	def __init__(self, message):
		self.message = message
	def __str__(self):
		return str(self.message)



class TimeEpochs(keras.callbacks.Callback):
	'''
	Callback used to calculate per-epoch time
	'''

	def on_train_begin(self, logs={}):
		self.times = []

	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)



class DLArchitectures(object):
	'''
	Implements state-of-the-art deep learning architectures.
	For details, refer to the related papers.
	'''

	def __init__(self, pickle_dataset_filenames, dataset, view, outdir, include_ports = True, is_multimodal = False):
		'''
		Constructor for the class DLArchitectures.
		Input:
		- pickle_dataset_filenames (list): TC datasets in .pickle format; for multimodal approaches len(pickle_dataset_filenames) > 1
		- dataset (int): set the type of dataset to handle the correct set of classes (dataset == 1 is FB/FBM; dataset == 2 is Android; dataset == 3 is iOS)
		- outdir (string): outdir to contain pickle dataset
		- include_ports (boolean): if True, TCP/UDP ports are used as input for lopez2017network approaches
		- is_multimodal (boolean): if True, a multimodal approach is considered and fed with multiple inputs (i.e. len(pickle_dataset_filenames) > 1)
		'''

		self.dataset = dataset
		self.view = view
		self.outdir = outdir

		self.K_binary = [1, 2]
		self.K_multi = [1, 3, 5, 7, 10]

		# TODO: Add K for multi-task

		self.gamma_range = numpy.arange(0.0, 1.0, 0.1)

		self.label_class_FB_FBM = {
							0: 'FB',
							1: 'FBM'
							}

		self.label_class_Android = {
							0: '360Security',
							1: '6Rooms',
							2: '80sMovie',
							3: '9YinZhenJing',
							4: 'Anghami',
							5: 'BaiDu',
							6: 'Crackle',
							7: 'EFood',
							8: 'FrostWire',
							9: 'FsecureFreedomeVPN',
							10: 'Go90',
							11: 'Google+',
							12: 'GoogleAllo',
							13: 'GoogleCast',
							14: 'GoogleMap',
							15: 'GooglePhotos',
							16: 'GooglePlay',
							17: 'GroupMe',
							18: 'Guvera',
							19: 'Hangouts',
							20: 'HidemanVPN',
							21: 'Hidemyass',
							22: 'Hooq',
							23: 'HotSpot',
							24: 'IFengNews',
							25: 'InterVoip',
							26: 'LRR',
							27: 'MeinO2',
							28: 'Minecraft',
							29: 'Mobily',
							30: 'Narutom',
							31: 'NetTalk',
							32: 'NileFM',
							33: 'Palringo',
							34: 'PaltalkScene',
							35: 'PrivateTunnelVPN',
							36: 'PureVPN',
							37: 'QQ',
							38: 'QQReader',
							39: 'QianXunYingShi',
							40: 'RaidCall',
							41: 'Repubblica',
							42: 'RiyadBank',
							43: 'Ryanair',
							44: 'SayHi',
							45: 'Shadowsocks',
							46: 'SmartVoip',
							47: 'Sogou',
							48: 'eBay'
		}

		self.label_class_IOS = {
							0: '360Security',
							1: '6Rooms',
							2: '80sMovie',
							3: 'Anghami',
							4: 'AppleiCloud',
							5: 'BaiDu',
							6: 'Brightcove',
							7: 'Crackle',
							8: 'EFood',
							9: 'FsecureFreedomeVPN',
							10: 'Go90',
							11: 'Google+',
							12: 'GoogleAllo',
							13: 'GoogleCast',
							14: 'GoogleMap',
							15: 'GooglePhotos',
							16: 'GroupMe',
							17: 'Guvera',
							18: 'Hangouts',
							19: 'HiTalk',
							20: 'HidemanVPN',
							21: 'Hidemyass',
							22: 'Hooq',
							23: 'HotSpot',
							24: 'IFengNews',
							25: 'LRR',
							26: 'MeinO2',
							27: 'Minecraft',
							28: 'Mobily',
							29: 'Narutom',
							30: 'NetTalk',
							31: 'NileFM',
							32: 'Palringo',
							33: 'PaltalkScene',
							34: 'PrivateTunnelVPN',
							35: 'PureVPN',
							36: 'QQReader',
							37: 'QianXunYingShi',
							38: 'Repubblica',
							39: 'Ryanair',
							40: 'SayHi',
							41: 'Shadowsocks',
							42: 'Sogou',
							43: 'eBay',
							44: 'iMessage'
		}

		self.label_class_VPN_nonVPN = {
			0: {
				0: 'VPN',
				1: 'nonVPN'
			},
			1: {
				0: 'Chat',
				1: 'Email',
				2: 'FileTransfer',
				3: 'P2P',
				4: 'Streaming',
				5: 'VoIP',
			},
			2: {
				0: 'aim',
				1: 'email',
				2: 'facebook',
				3: 'ftps',
				4: 'hangouts',
				5: 'icq',
				6: 'netflix',
				7: 'scp',
				8: 'sftp',
				9: 'skype',
				10: 'spotify',
				11: 'torrent',
				12: 'voipbuster',
				13: 'youtube'
			}
		}

		# TODO: Add label lists for multi-task

		self.time_callback = TimeEpochs()

		if self.dataset == 1:
			self.K = self.K_binary
			self.label_class = self.label_class_FB_FBM
		elif self.dataset == 2:
			self.K = self.K_multi
			self.label_class = self.label_class_Android
		elif self.dataset == 3:
			self.K = self.K_multi
			self.label_class = self.label_class_IOS
		elif self.dataset == 4:
			if self.view == 1:
				self.K = self.K_binary
				self.label_class = self.label_class_VPN_nonVPN[self.view-1]
			elif self.view == 2:
				self.K = [1,2,3]
				self.label_class = self.label_class_VPN_nonVPN[self.view-1]
			elif self.view == 3:
				self.K = self.K_multi
				self.label_class = self.label_class_VPN_nonVPN[self.view-1]



		self.debug_log = self.setup_logger('debug', '%s/deep_learning_architectures.log' % self.outdir, logging.DEBUG)

		for pickle_dataset_filename in pickle_dataset_filenames:
			if os.path.isfile(pickle_dataset_filename):
				self.debug_log.debug('The dataset file %s exists.' % pickle_dataset_filename)
			else:
				err_str = 'The dataset file %s does not exist. Please, provide a valid .pickle file.' % pickle_dataset_filename
				self.debug_log.error(err_str)
				raise DLArchitecturesException(err_str)

		self.pickle_dataset_filenames = pickle_dataset_filenames
		self.deserialize_dataset(include_ports, is_multimodal, self.view)


	def setup_logger(self, logger_name, log_file, level=logging.INFO):
		'''
		Return a log object associated to <log_file> with specific formatting and rotate handling.
		'''

		l = logging.getLogger(logger_name)

		if not getattr(l, 'handler_set', None):
			logging.Formatter.converter = time.gmtime
			formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

			rotatehandler = RotatingFileHandler(log_file, mode='a', maxBytes=10485760, backupCount=30)
			rotatehandler.setFormatter(formatter)

			#TODO: Debug stream handler for development
			#streamHandler = logging.StreamHandler()
			#streamHandler.setFormatter(formatter)
			#l.addHandler(streamHandler)

			l.addHandler(rotatehandler)
			l.setLevel(level)
			l.handler_set = True

		elif not os.path.exists(log_file):
			logging.Formatter.converter = time.gmtime
			formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

			rotatehandler = RotatingFileHandler(log_file, mode='a', maxBytes=10485760, backupCount=30)
			rotatehandler.setFormatter(formatter)

			#TODO: Debug stream handler for development
			#streamHandler = logging.StreamHandler()
			#streamHandler.setFormatter(formatter)
			#l.addHandler(streamHandler)

			l.addHandler(rotatehandler)
			l.setLevel(level)
			l.handler_set = True

		return l


	def wang2017endtoend_1DCNN(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the 1D Convolutional Neural Network proposed in wang2017endtoend
		'''

		self.debug_log.info('Starting execution of wang2017endtoend_1DCNN approach')

		# Define a Sequential model
		model = Sequential()

		# Build 1D-CNN, according to wang2017endtoend paper
		model.add(Conv1D(filters = 32, kernel_size = 25, strides = 1, padding = 'same', activation = 'relu', input_shape = (self.input_dim, 1)))
		model.add(MaxPooling1D(pool_size = 3, strides = None, padding = 'same'))
		model.add(Conv1D(filters = 64, kernel_size = 25, strides = 1, padding = 'same', activation = 'relu'))
		model.add(MaxPooling1D(pool_size = 3, strides = None, padding = 'same'))
		model.add(Flatten())
		model.add(Dense(1024, activation = 'relu'))
		# model.add(Dropout(0.2))
		model.add(Dense(self.num_classes, activation = 'softmax'))
		# model.summary()

		self.debug_log.info('wang2017endtoend_1DCNN: Sequential model built')

		# Define SGD optimizerextract
		# sgd_opt = optimizers.SGD(lr = 1e-4)

		# Compile the model with categorical crossentropy loss function
		# model.compile(loss='categorical_crossentropy', optimizer=sgd_opt, metrics=['accuracy'])

		# SGD optimizer with categorical crossentropy loss function and default learning rate
		model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

		self.debug_log.info('wang2017endtoend_1DCNN: Sequential model compiled with crossentropy loss function')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/wang2017endtoend_1D-CNN.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0.01, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 60, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('wang2017endtoend_1DCNN: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('wang2017endtoend_1DCNN: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of wang2017endtoend_1DCNN approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def wang2017malware_2DCNN(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the 2D Convolutional Neural Network proposed in wang2017malware
		'''

		self.debug_log.info('Starting execution of wang2017malware_2DCNN approach')

		input_dim_sqrt = int(numpy.sqrt(self.input_dim))

		# Define a Sequential model
		model = Sequential()

		# Build 2D-CNN, according to wang2017malware paper
		model.add(Reshape((input_dim_sqrt, input_dim_sqrt, 1), input_shape = (self.input_dim, 1)))
		model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
		model.add(Dropout(0.5))
		model.add(MaxPooling2D(pool_size = (2, 2), strides = None, padding = 'same'))
		model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
		model.add(MaxPooling2D(pool_size = (2, 2), strides = None, padding = 'same'))
		model.add(Flatten())
		model.add(Dense(1024, activation = 'relu'))
		# model.add(Dropout(0.2))
		model.add(Dense(self.num_classes, activation = 'softmax'))
		# model.summary()

		self.debug_log.info('wang2017malware_2DCNN: Sequential model built')

		# Compile the model with categorical crossentropy loss function
		model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

		self.debug_log.info('wang2017malware_2DCNN: Sequential model compiled with crossentropy loss function')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/wang2017malware_2D-CNN.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0.01, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 60, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('wang2017malware_2DCNN: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('wang2017malware_2DCNN: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of wang2017malware_2DCNN approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	# TODO: Network parameters should be verified
	def lotfollahi2017deep_1DCNN(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the 1D Convolutional Neural Network proposed in lotfollahi2017deep
		'''

		self.debug_log.info('Starting execution of lotfollahi2017deep_1DCNN approach')

		# Define a Sequential model
		model = Sequential()

		# Build 1D-CNN, according to lotfollahi2017deep paper
		model.add(Conv1D(filters = 20, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu', input_shape = (self.input_dim, 1)))
		model.add(Conv1D(filters = 8, kernel_size = 4, strides = 1, padding = 'same', activation = 'relu'))
		model.add(AveragePooling1D(pool_size = 2, strides = None, padding = 'same'))
		model.add(Dropout(0.25))
		model.add(Flatten())
		# 7 Full Connected layers
		model.add(Dense(60, activation = 'relu'))
		model.add(Dense(50, activation = 'relu'))
		model.add(Dense(40, activation = 'relu'))
		model.add(Dense(30, activation = 'relu'))
		model.add(Dense(20, activation = 'relu'))
		model.add(Dense(10, activation = 'relu'))
		model.add(Dense(50, activation = 'relu'))
		model.add(Dense(self.num_classes, activation = 'softmax'))
		# model.summary()

		self.debug_log.info('lotfollahi2017deep_1DCNN: Sequential model built')

		# Compile the model with categorical crossentropy loss function and Adam optimizer
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		self.debug_log.info('lotfollahi2017deep_1DCNN: Sequential model compiled with crossentropy loss function and Adam optimizer')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/lotfollahi2017deep_1D-CNN.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 20, 100, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
		self.debug_log.info('lotfollahi2017deep_1DCNN: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('lotfollahi2017deep_1DCNN: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of lotfollahi2017deep_1DCNN approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def lotfollahi2017deep_SAE(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the SAE architecture proposed in lotfollahi2017deep
		'''

		self.debug_log.info('Starting execution of lotfollahi2017deep_SAE approach')

		# Stacked Autoencoders (no denoising)
		sdae = StackedDenoisingAutoencoder(layers=[400, 300, 200, 100, 50], do_pretrain=True, \
						   enc_act_func=[tf.nn.sigmoid], dec_act_func=[tf.nn.sigmoid], \
						   opt = ['sgd'], num_epochs = [200], batch_size = [50], \
						   finetune_act_func=tf.nn.relu, finetune_opt='adam', finetune_loss_func='softmax_cross_entropy', \
						   finetune_learning_rate=0.0001, finetune_num_epochs=200, finetune_batch_size=50, finetune_dropout=0.75)

		self.debug_log.info('lotfollahi2017deep_SAE: StackedDenoisingAutoencoder model built')

		# *** TRAINING PHASE ***
		self.train_SAE(sdae, samples_train, categorical_labels_train, self.num_classes)

		# *** TEST PHASE ***
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_SAE(sdae, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('lotfollahi2017deep_SAE: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of lotfollahi2017deep_SAE approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def lopez2017network_CNN_1(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the CNN-1 architecture proposed in lopez2017network
		'''

		self.debug_log.info('Starting execution of lopez2017network_CNN_1 approach')

		try:
			# Define a Sequential model
			model = Sequential()

			# Build 2D-CNN (CNN-1), according to lopez2017network paper
			model.add(Conv2D(filters = 32, kernel_size = (4, 2), strides = 1, padding = 'valid', activation='relu', input_shape = (self.input_dim, self.input_dim_2, 1)))
			model.add(MaxPooling2D(pool_size = (3, 2), strides = 1, padding = 'valid'))
			model.add(BatchNormalization())
			model.add(Conv2D(filters = 64, kernel_size = (4, 2), strides = 1, padding = 'valid', activation='relu'))
			if self.input_dim_2 >= 6:
				model.add(MaxPooling2D(pool_size = (3, 2), strides = 1, padding = 'valid'))
			else:
				model.add(MaxPooling2D(pool_size = (3, 1), strides = 1, padding = 'valid'))
			model.add(BatchNormalization())
			model.add(Flatten())
			model.add(Dense(200, activation = 'relu'))
			model.add(Dense(self.num_classes, activation = 'softmax'))
			# model.summary()

		except ValueError:
			# Define a Sequential model
			model = Sequential()

			# Build 2D-CNN (CNN-1), according to lopez2017network when the number of packets is less than 12
			model.add(Conv2D(filters = 32, kernel_size = (4, 2), strides = 1, padding = 'same', activation='relu', input_shape = (self.input_dim, self.input_dim_2, 1)))
			model.add(MaxPooling2D(pool_size = (3, 2), strides = 1, padding = 'same'))
			model.add(BatchNormalization())
			model.add(Conv2D(filters = 64, kernel_size = (4, 2), strides = 1, padding = 'same', activation='relu'))
			if self.input_dim_2 >= 6:
				model.add(MaxPooling2D(pool_size = (3, 2), strides = 1, padding = 'same'))
			else:
				model.add(MaxPooling2D(pool_size = (3, 1), strides = 1, padding = 'same'))
			model.add(BatchNormalization())
			model.add(Flatten())
			model.add(Dense(200, activation = 'relu'))
			model.add(Dense(self.num_classes, activation = 'softmax'))
			# model.summary()

		self.debug_log.info('lopez2017network_CNN_1: Sequential model built')

		# Compile the model with categorical crossentropy loss function and Adam optimizer
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		self.debug_log.info('lopez2017network_CNN_1: Sequential model compiled with crossentropy loss function and Adam optimizer')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/lopez2017network_CNN-1.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 90, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('lopez2017network_CNN_1: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('lopez2017network_CNN_1: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of lopez2017network_CNN_1 approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def lopez2017network_RNN_1(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the RNN-1 architecture proposed in lopez2017network
		'''

		self.debug_log.info('Starting execution of lopez2017network_RNN_1 approach')

		# Define a Sequential model
		model = Sequential()

		# Build LSTM (RNN-1), according to lopez2017network paper
		model.add(LSTM(100, input_shape = (self.input_dim, self.input_dim_2)))
		model.add(Dense(100, activation = 'relu'))
		model.add(Dense(self.num_classes, activation = 'softmax'))
		# model.summary()

		self.debug_log.info('lopez2017network_RNN_1: Sequential model built')

		# Compile the model with categorical crossentropy loss function and Adam optimizer
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		self.debug_log.info('lopez2017network_RNN_1: Sequential model compiled with crossentropy loss function and Adam optimizer')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/lopez2017network_RNN-1.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 90, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('lopez2017network_RNN_1: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('lopez2017network_RNN_1: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of lopez2017network_RNN_1 approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def lopez2017network_CNN_RNN_2a(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the CNN+RNN-2a architecture proposed in lopez2017network
		'''

		self.debug_log.info('Starting execution of lopez2017network_CNN_RNN_2a approach')

		try:
			# Define a Sequential model
			model = Sequential()

			# Build CNN+LSTM (CNN_RNN_2a), according to lopez2017network paper
			model.add(Conv2D(filters = 32, kernel_size = (4, 2), strides = 1, padding = 'valid', activation='relu', input_shape = (self.input_dim, self.input_dim_2, 1)))
			model.add(BatchNormalization())
			model.add(Conv2D(filters = 64, kernel_size = (4, 2), strides = 1, padding = 'valid', activation='relu'))
			model.add(BatchNormalization())
			shape_interm = model.layers[3].output_shape
			model.add(Reshape((shape_interm[2], shape_interm[3] * shape_interm[1])))
			model.add(LSTM(100))
			model.add(Dropout(0.2))
			model.add(Dense(100, activation = 'relu'))
			model.add(Dropout(0.4))
			model.add(Dense(self.num_classes, activation = 'softmax'))
			# model.summary()

		except ValueError:
			# Define a Sequential model
			model = Sequential()

			# Build CNN+LSTM (CNN_RNN_2a), according to lopez2017network paper
			model.add(Conv2D(filters = 32, kernel_size = (4, 2), strides = 1, padding = 'same', activation='relu', input_shape = (self.input_dim, self.input_dim_2, 1)))
			model.add(BatchNormalization())
			model.add(Conv2D(filters = 64, kernel_size = (4, 2), strides = 1, padding = 'same', activation='relu'))
			model.add(BatchNormalization())
			shape_interm = model.layers[3].output_shape
			model.add(Reshape((shape_interm[2], shape_interm[3] * shape_interm[1])))
			model.add(LSTM(100))
			model.add(Dropout(0.2))
			model.add(Dense(100, activation = 'relu'))
			model.add(Dropout(0.4))
			model.add(Dense(self.num_classes, activation = 'softmax'))
			# model.summary()

		self.debug_log.info('lopez2017network_CNN_RNN_2a: Sequential model built')

		# Compile the model with categorical crossentropy loss function and Adam optimizer
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		self.debug_log.info('lopez2017network_CNN_RNN_2a: Sequential model compiled with crossentropy loss function and Adam optimizer')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/lopez2017network_CNN_RNN-2a.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 90, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('lopez2017network_CNN_RNN_2a: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('lopez2017network_CNN_RNN_2a: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of lopez2017network_CNN_RNN_2a approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def oh2017traffic_2DCNN(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the 2D Convolutional Neural Network proposed in oh2017traffic
		'''

		self.debug_log.info('Starting execution of oh2017traffic_2DCNN approach')

		input_dim_sqrt = int(numpy.sqrt(self.input_dim))

		# Define a Sequential model
		model = Sequential()

		# Build 2D-CNN, according to oh2017traffic paper
		model.add(Reshape((input_dim_sqrt, input_dim_sqrt, 1), input_shape = (self.input_dim, 1)))
		model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.01)))
		model.add(MaxPooling2D(pool_size = (2, 2), strides = None, padding = 'same'))
		model.add(Flatten())
		model.add(Dense(256, activation = 'relu'))
		model.add(Dropout(0.8))
		model.add(Dense(self.num_classes, activation = 'softmax'))
		# model.summary()

		self.debug_log.info('oh2017traffic_2DCNN: Sequential model built')

		# Compile the model with categorical crossentropy loss function and adam optimizer
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		self.debug_log.info('oh2017traffic_2DCNN: Sequential model compiled with crossentropy loss function')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/oh2017traffic_2D-CNN.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0.01, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 300, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('oh2017traffic_2DCNN: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('oh2017traffic_2DCNN: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of oh2017traffic_2DCNN approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def oh2017traffic_MLP(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the Multi Layer Perceptron proposed in oh2017traffic
		'''

		self.debug_log.info('Starting execution of oh2017traffic_MLP approach')

		shallow_nodes = 256

		# Define a Sequential model
		model = Sequential()

		# 3-layers oh2017traffic_MLP
		model.add(Dense(shallow_nodes, activation='tanh', input_dim = self.input_dim, kernel_regularizer = l2(0.01)))
		model.add(Dropout(0.8))
		model.add(Dense(shallow_nodes, activation='tanh', kernel_regularizer = l2(0.01)))
		model.add(Dropout(0.8))
		model.add(Dense(self.num_classes, activation = 'softmax'))
		# model.summary()

		self.debug_log.info('oh2017traffic_MLP: Sequential model built')

		# SGD optimizer with categorical crossentropy loss function and default learning rate
		model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

		self.debug_log.info('oh2017traffic_MLP: Sequential model compiled with crossentropy loss function')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/oh2017traffic_MLP.png" % self.outdir))

		# Define early stopping and TimeEpochs callback
		earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 500, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('oh2017traffic_MLP: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('oh2017traffic_MLP: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of oh2017traffic_MLP approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	# XXX: Simple SAE used only for test purpose
	def wang2015applications_SAE(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the SAE architecture proposed in wang2015applications
		'''

		self.debug_log.info('Starting execution of wang2015applications_SAE approach')

		# Stacked Autoencoders (no denoising)
		sdae = StackedDenoisingAutoencoder(layers=[100, 50, 40, 30], do_pretrain=True, corr_type="none", \
						   enc_act_func=[tf.nn.sigmoid], dec_act_func=[tf.nn.sigmoid], \
						   num_epochs = [60], batch_size = [50], \
						   finetune_learning_rate=0.01, finetune_num_epochs=60, finetune_batch_size=50)

		self.debug_log.info('wang2015applications_SAE: StackedDenoisingAutoencoder model built')

		# *** TRAINING PHASE ***
		self.train_SAE(sdae, samples_train, categorical_labels_train, self.num_classes)

		# *** TEST PHASE ***
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_SAE(sdae, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('wang2015applications_SAE: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of wang2015applications_SAE approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def single_layer_perceptron(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements a single-layer perceptron used as a baseline for DL-networks evaluation
		'''

		self.debug_log.info('Starting execution of single_layer_perceptron approach')

		shallow_nodes = 100

		# Define a Sequential model
		model = Sequential()

		# Build a single-layer perceptron
		model.add(Dense(shallow_nodes, activation = 'relu', input_dim = self.input_dim))
		model.add(Dense(self.num_classes, activation = 'softmax'))
		# model.summary()

		self.debug_log.info('single_layer_perceptron: Sequential model built')

		# Compile the model with categorical crossentropy loss function
		# model.compile(loss='categorical_crossentropy', optimizer=sgd_opt, metrics=['accuracy'])

		# Compile the model with categorical crossentropy loss function and Adam optimizer
		model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

		self.debug_log.info('single_layer_perceptron: Sequential model compiled with crossentropy loss function')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/single_layer_perceptron.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 60, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('single_layer_perceptron: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('single_layer_perceptron: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of single_layer_perceptron approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def taylor2016appscanner_RF(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the RF proposed in taylor2016appscanner used as a baseline for DL-networks evaluation
		'''

		self.debug_log.info('Starting execution of taylor2016appscanner_RF approach')

		transformer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, verbose=2, copy=True)
		transformed_samples_train = transformer.fit_transform(samples_train)
		transformed_samples_test = transformer.transform(samples_test)

		estimators_number = 150

		# Define RF classifier with parameters specified in taylor2016appcanner
		clf = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')

		# Training phase
		train_time_begin = time.time()
		clf.fit(transformed_samples_train, categorical_labels_train)
		train_time_end = time.time()
		self.train_time = train_time_end - train_time_begin
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, transformed_samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_classifier(clf, transformed_samples_train, categorical_labels_train, transformed_samples_test, categorical_labels_test)
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def single_layer_perceptron_lopez(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements a single-layer perceptron fed with lopez2017network input data used as a baseline for DL-networks evaluation
		'''

		self.debug_log.info('Starting execution of single_layer_perceptron_lopez approach')

		shallow_nodes = 100

		# Define a Sequential model
		model = Sequential()

		# Build a single-layer perceptron compatible with lopez2017network input data
		model.add(Flatten(input_shape = (self.input_dim, self.input_dim_2)))
		model.add(Dense(shallow_nodes, activation = 'relu'))
		model.add(Dense(self.num_classes, activation = 'softmax'))
		# model.summary()

		self.debug_log.info('single_layer_perceptron_lopez: Sequential model built')

		# Compile the model with categorical crossentropy loss function and Adam optimizer
		model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

		self.debug_log.info('single_layer_perceptron_lopez: Sequential model compiled with crossentropy loss function')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/single_layer_perceptron_lopez.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 60, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('single_layer_perceptron_lopez: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('single_layer_perceptron_lopez: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of single_layer_perceptron_lopez approach')
		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU(self, samples_train_list, categorical_labels_train, samples_test_list, categorical_labels_test):
		'''
		Implements a multimodal architecture based on approaches proposed in wang2017endtoend and lopez2017network
		'''

		self.debug_log.info('Starting execution of multimodal_wang2017endtoend_lopez2017network approach')

		if len(self.input_dims_list) != 2:
			err_str = 'multimodal_wang2017endtoend_lopez2017network: provided datasets are not two.'
			self.debug_log.error(err_str)
			raise DLArchitecturesException(err_str)

		for i, input_dim in enumerate(self.input_dims_list):
			try:
				wang_input_dim = int(input_dim)
				samples_wang_train = numpy.expand_dims(samples_train_list[i], axis=2)
				samples_wang_test = numpy.expand_dims(samples_test_list[i], axis=2)
			except TypeError:
				lopez_input_dim = input_dim
				samples_lopez_train = samples_train_list[i]
				samples_lopez_test = samples_test_list[i]

		num_samples_c0 = numpy.shape(samples_wang_train[categorical_labels_train == 0, :])[0]
		num_samples_c1 = numpy.shape(samples_wang_train[categorical_labels_train == 1, :])[0]

		# print(num_samples_c0)
		# print(num_samples_c1)

		scaler = QuantileTransformer()
		for j in range(lopez_input_dim[0]):
			scaler.fit(samples_lopez_train[:,j,:])
			samples_lopez_train[:,j,:] = scaler.transform(samples_lopez_train[:,j,:])
			samples_lopez_test[:,j,:] = scaler.transform(samples_lopez_test[:,j,:])

		# Define a Sequential model
		multimodal_model = Sequential()

		# Define input types to fed to multimodal model
		wang_payload = Input(shape = (wang_input_dim, 1), name = 'wang_payload')
		lopez_mat_fields = Input(shape = lopez_input_dim, name = 'lopez_packetfields')

		# Build 1D-CNN, according to wang2017endtoend paper
		w = Dropout(0.2) (wang_payload)
		w = Conv1D(filters = 16, kernel_size = 25, strides = 1, kernel_constraint = maxnorm(3), padding = 'valid', activation = 'relu') (w)
		w = MaxPooling1D(pool_size = 3, strides = None, padding = 'valid') (w)
		w = Conv1D(filters = 32, kernel_size = 25, strides = 1, kernel_constraint = maxnorm(3), padding = 'valid', activation = 'relu') (w)
		w = MaxPooling1D(pool_size = 3, strides = None, padding = 'valid') (w)
		w = Flatten() (w)
		# w = BatchNormalization() (w)
		w = Dropout(0.2) (w)
		interm_wang = Dense(100, activation = 'relu') (w)

		# Build a bidirectional GRU fed with the input proposed in lopez2017network
		l = Bidirectional(GRU(50, return_sequences = True, activation = 'relu', kernel_constraint = maxnorm(3))) (lopez_mat_fields)
		l = Flatten() (l)
		l = Dropout(0.2) (l)
		# l = Dense(30, activation = 'relu') (l)
		interm_lopez = Dense(100, activation = 'relu', kernel_constraint = maxnorm(3)) (l)

		y = concatenate([interm_wang, interm_lopez])
		# y = BatchNormalization() (y)
		y = Dropout(0.2) (y)
		# y = Dense(40, activation = 'relu', kernel_constraint = maxnorm(3)) (y)
		# y = Dropout(0.2) (y)
		output_multimodal = Dense(self.num_classes, kernel_constraint = maxnorm(3), activation = 'softmax') (y)

		multimodal_model = Model(inputs = [wang_payload, lopez_mat_fields], outputs = output_multimodal)

		self.debug_log.info('multimodal_wang2017endtoend_lopez2017network: Model() model built')

		# Compile the model with categorical crossentropy loss function and Adam optimizer
		multimodal_model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])

		self.debug_log.info('multimodal_wang2017endtoend_lopez2017network: Model() model compiled with crossentropy loss function')

		# Plot DL architecture in PNG format
		plot_model(multimodal_model, to_file=("%s/multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta = 0.01, patience=15, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_multimodal_model(multimodal_model, epochs, batch_size, callbacks, samples_train_list, categorical_labels_train, num_classes, num_samples_c0, num_samples_c1)
		self.train_multimodal_model(multimodal_model, 90, 50, callbacks_list, [samples_wang_train, samples_lopez_train], \
		categorical_labels_train, self.num_classes, num_samples_c0, num_samples_c1)
		self.debug_log.info('multimodal_wang2017endtoend_lopez2017network: Training phase completed')
		print('Training phase completed')

		# Test with predict: test_multimodal_model(multimodal_model, samples_train_list, categorical_labels_train, samples_test_list, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_multimodal_model(multimodal_model, [samples_wang_train, samples_lopez_train], categorical_labels_train, [samples_wang_test, samples_lopez_test], categorical_labels_test)
		self.debug_log.info('multimodal_wang2017endtoend_lopez2017network: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of multimodal_wang2017endtoend_lopez2017network approach')
		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	# XXX: Only for test purpose
	def multi_layer_perceptron_test_lotfollahi2017deep_SAE(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements a multi-layer perceptron with the same layers as lotfollahi2017deep_SAE
		'''

		self.debug_log.info('Starting execution of multi_layer_perceptron_test_lotfollahi2017deep_SAE approach')

		# Define a Sequential model
		model = Sequential()

		# Build a single-layer perceptron
		model.add(Dense(400, activation = 'relu', input_dim = self.input_dim))
		model.add(Dense(300, activation = 'relu'))
		model.add(Dense(200, activation = 'relu'))
		model.add(Dense(100, activation = 'relu'))
		model.add(Dense(50, activation = 'relu'))
		model.add(Dense(self.num_classes, activation = 'softmax'))
		# model.summary()

		self.debug_log.info('multi_layer_perceptron_test_lotfollahi2017deep_SAE: Sequential model built')

		# Compile the model with categorical crossentropy loss function
		# model.compile(loss='categorical_crossentropy', optimizer=sgd_opt, metrics=['accuracy'])

		# Compile the model with categorical crossentropy loss function and Adam optimizer
		model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

		self.debug_log.info('multi_layer_perceptron_test_lotfollahi2017deep_SAE: Sequential model compiled with crossentropy loss function')

		# Plot DL architecture in PNG format
		plot_model(model, to_file=("%s/multi_layer_perceptron_test_lotfollahi2017deep_SAE.png" % self.outdir))

		# Define early stopping and TimeEpochs callbacks
		earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
		callbacks_list = [self.time_callback, earlystop]

		# Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
		self.train_model(model, 200, 50, callbacks_list, samples_train, categorical_labels_train, self.num_classes)
		self.debug_log.info('multi_layer_perceptron_test_lotfollahi2017deep_SAE: Training phase completed')
		print('Training phase completed')

		# Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
		predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
		self.debug_log.info('multi_layer_perceptron_test_lotfollahi2017deep_SAE: Test phase completed')
		print('Test phase completed')

		self.debug_log.info('Ending execution of multi_layer_perceptron_test_lotfollahi2017deep_SAE approach')

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios


	def stratified_cross_validation(self, dl_model, num_fold, seed, do_expand_dims=True, do_normalization=False, is_multimodal=False):
		'''
		Perform a stratified cross validation of the state-of-the-art DL (multimodal) model given as input
		'''

		self.debug_log.info('Starting execution of stratified cross validation')

		with open('%s/performance/%s_performance.dat' % (self.outdir, dl_model.__name__), 'w') as performance_file:
			performance_file.write('%s\tAccuracy\tMacro_F-measure\tG-mean\tMacro_G-mean\n' % dl_model.__name__)

		with open('%s/performance/%s_training_performance.dat' % (self.outdir, dl_model.__name__), 'w') as training_performance_file:
			training_performance_file.write('%s\tTraining_Accuracy\tTraining_Macro_F-measure\tTraining_G-mean\tTraining_Macro_G-mean\n' % dl_model.__name__)

		with open('%s/performance/%s_top-k_accuracy.dat' % (self.outdir, dl_model.__name__), 'w') as topk_accuracy_file:
			topk_accuracy_file.write('%s\t' % dl_model.__name__)
			for k in self.K:
				topk_accuracy_file.write('K=%d\t' % k)
			topk_accuracy_file.write('\n')

		with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'w') as filtered_perfomance_file:
			filtered_perfomance_file.write('%s\t' % dl_model.__name__)
			for gamma in self.gamma_range:
				filtered_perfomance_file.write('c=%f\t' % gamma)
			filtered_perfomance_file.write('\n')

		if dl_model.__name__ != "wang2015applications_SAE" and dl_model.__name__ != "lotfollahi2017deep_SAE":
			time_epochs_filename = '%s/performance/%s_per_epoch_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_epochs_filename, 'w') as time_epochs_file:
				time_epochs_file.write('%s\tEpoch_times\n' % dl_model.__name__)

		if dl_model.__name__ == "wang2015applications_SAE" or dl_model.__name__ == "lotfollahi2017deep_SAE":
			time_sdae_filename = '%s/performance/%s_sdae_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_sdae_filename, 'w') as time_sdae_file:
				time_sdae_file.write('%s\tPre-training_time\tTraining_time\n' % dl_model.__name__)

		if dl_model.__name__ == "taylor2016appscanner_RF":
			time_RF_filename = '%s/performance/%s_RF_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_RF_filename, 'w') as time_RF_file:
				time_RF_file.write('%s\tTraining_time\n' % dl_model.__name__)

		test_time_filename = '%s/performance/%s_test_times.dat' % (self.outdir, dl_model.__name__)
		with open(test_time_filename, 'w') as test_time_file:
				test_time_file.write('%s\tTest_time\n' % dl_model.__name__)

		cvscores_pred = []
		fscores_pred = []
		gmeans_pred = []
		macro_gmeans_pred = []
		training_cvscores_pred = []
		training_fscores_pred = []
		training_gmeans_pred = []
		training_macro_gmeans_pred = []
		norm_cms_pred = []
		topk_accuracies_pred = []
		filtered_cvscores_pred = []
		filtered_fscores_pred = []
		filtered_gmeans_pred = []
		filtered_macro_gmeans_pred = []
		filtered_norm_cms_pred = []
		filtered_classified_ratios_pred = []

		current_fold = 1
		kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed)

		for train, test in kfold.split(self.samples, self.categorical_labels):
			self.debug_log.info('stratified_cross_validation: Starting fold %d execution' % current_fold)

			if is_multimodal:
				# In the multimodal approaches, preprocessing operations strongly depend on the specific DL architectures and inputs considered.
				samples_train_list = []
				samples_test_list = []

				for dataset_samples in self.samples_list:
					samples_train_list.append(dataset_samples[train])
					samples_test_list.append(dataset_samples[test])

				predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
				filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
				filtered_norm_cnf_matrices, filtered_classified_ratios = \
				dl_model(samples_train_list, self.categorical_labels[train], samples_test_list, self.categorical_labels[test])

			else:
				if do_normalization and (dl_network == 4 or dl_network == 5 or dl_network == 6 or dl_network == 11):
					scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
					scaler.fit(numpy.reshape(self.samples[train], [-1, self.input_dim_2]))
					res_samples_train = scaler.transform(numpy.reshape(self.samples[train], [-1, self.input_dim_2]))
					res_samples_test = scaler.transform(numpy.reshape(self.samples[test], [-1, self.input_dim_2]))
					samples_train = numpy.reshape(res_samples_train, [-1, self.input_dim, self.input_dim_2])
					samples_test = numpy.reshape(res_samples_test, [-1, self.input_dim, self.input_dim_2])
				else:
					samples_train = self.samples[train]
					samples_test = self.samples[test]

				if do_expand_dims:
					if dl_network == 4 or dl_network == 6 or dl_network == 11:
						# CNN-based approaches implemented in lopez2017network
						samples_train = numpy.expand_dims(samples_train, axis=3)
						samples_test = numpy.expand_dims(samples_test, axis=3)
					else:
						samples_train = numpy.expand_dims(samples_train, axis=2)
						samples_test = numpy.expand_dims(samples_test, axis=2)

				predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
				filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
				filtered_norm_cnf_matrices, filtered_classified_ratios = \
				dl_model(samples_train, self.categorical_labels[train], samples_test, self.categorical_labels[test])

			accuracy = accuracy * 100
			fmeasure = fmeasure * 100
			gmean = gmean * 100
			macro_gmean = macro_gmean * 100
			training_accuracy = training_accuracy * 100
			training_fmeasure = training_fmeasure * 100
			training_gmean = training_gmean * 100
			training_macro_gmean = training_macro_gmean * 100
			norm_cnf_matrix = norm_cnf_matrix * 100
			topk_accuracies[:] = [topk_acc * 100 for topk_acc in topk_accuracies]
			filtered_accuracies[:] = [filtered_accuracy * 100 for filtered_accuracy in filtered_accuracies]
			filtered_fmeasures[:] = [filtered_fmeasure * 100 for filtered_fmeasure in filtered_fmeasures]
			filtered_gmeans[:] = [filtered_gmean * 100 for filtered_gmean in filtered_gmeans]
			filtered_macro_gmeans[:] = [filtered_macro_gmean * 100 for filtered_macro_gmean in filtered_macro_gmeans]
			filtered_norm_cnf_matrices[:] = [filtered_norm_cnf_matrix * 100 for filtered_norm_cnf_matrix in filtered_norm_cnf_matrices]
			filtered_classified_ratios[:] = [filtered_classified_ratio * 100 for filtered_classified_ratio in filtered_classified_ratios]

			sorted_labels = sorted(set(self.categorical_labels))

			cvscores_pred.append(accuracy)
			fscores_pred.append(fmeasure)
			gmeans_pred.append(gmean)
			macro_gmeans_pred.append(macro_gmean)
			training_cvscores_pred.append(training_accuracy)
			training_fscores_pred.append(training_fmeasure)
			training_gmeans_pred.append(training_gmean)
			training_macro_gmeans_pred.append(training_macro_gmean)
			norm_cms_pred.append(norm_cnf_matrix)
			topk_accuracies_pred.append(topk_accuracies)
			filtered_cvscores_pred.append(filtered_accuracies)
			filtered_fscores_pred.append(filtered_fmeasures)
			filtered_gmeans_pred.append(filtered_gmeans)
			filtered_macro_gmeans_pred.append(filtered_macro_gmeans)
			filtered_norm_cms_pred.append(filtered_norm_cnf_matrices)
			filtered_classified_ratios_pred.append(filtered_classified_ratios)

			# Evaluation of performance metrics
			print("Fold %d accuracy: %.4f%%" % (current_fold, accuracy))
			print("Fold %d macro F-measure: %.4f%%" % (current_fold, fmeasure))
			print("Fold %d g-mean: %.4f%%" % (current_fold, gmean))
			print("Fold %d macro g-mean: %.4f%%" % (current_fold, macro_gmean))

			with open('%s/performance/%s_performance.dat' % (self.outdir, dl_model.__name__), 'a') as performance_file:
				performance_file.write('Fold_%d\t%.4f%%\t%.4f%%\t%.4f%%\t%.4f%%\n' % (current_fold, accuracy, fmeasure, gmean, macro_gmean))

			# Evaluation of performance metrics on training set
			print("Fold %d training accuracy: %.4f%%" % (current_fold, training_accuracy))
			print("Fold %d training macro F-measure: %.4f%%" % (current_fold, training_fmeasure))
			print("Fold %d training g-mean: %.4f%%" % (current_fold, training_gmean))
			print("Fold %d training macro g-mean: %.4f%%" % (current_fold, training_macro_gmean))

			with open('%s/performance/%s_training_performance.dat' % (self.outdir, dl_model.__name__), 'a') as training_performance_file:
				training_performance_file.write('Fold_%d\t%.4f%%\t%.4f%%\t%.4f%%\t%.4f%%\n' % (current_fold, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean))

			# Confusion matrix
			print("Fold %d confusion matrix:" % current_fold)
			for i, row in enumerate(norm_cnf_matrix):
				for occurences in row:
					sys.stdout.write('%s,' % occurences)
				sys.stdout.write('%s\n' % sorted_labels[i])

			# Top-K accuracy
			with open('%s/performance/%s_top-k_accuracy.dat' % (self.outdir, dl_model.__name__), 'a') as topk_accuracy_file:
				topk_accuracy_file.write('Fold_%d' % current_fold)
				for i, topk_accuracy in enumerate(topk_accuracies):
					print("Fold %d top-%d accuracy: %.4f%%" % (current_fold, i, topk_accuracy))
					topk_accuracy_file.write('\t%.4f' % topk_accuracy)
				topk_accuracy_file.write('\n')

			# Accuracy, F-measure, and g-mean with reject option
			with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
				sys.stdout.write('Fold %d filtered accuracy:' % current_fold)
				filtered_perfomance_file.write('Fold_%d_filtered_accuracy' % current_fold)
				for filtered_accuracy in filtered_accuracies:
					sys.stdout.write('\t%.4f%%' % filtered_accuracy)
					filtered_perfomance_file.write('\t%.4f' % filtered_accuracy)
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')

				sys.stdout.write('Fold %d filtered F-measure:' % current_fold)
				filtered_perfomance_file.write('Fold_%d_filtered_F-measure' % current_fold)
				for filtered_fmeasure in filtered_fmeasures:
					sys.stdout.write('\t%.4f%%' % filtered_fmeasure)
					filtered_perfomance_file.write('\t%.4f' % filtered_fmeasure)
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')

				sys.stdout.write('Fold %d filtered g-mean:' % current_fold)
				filtered_perfomance_file.write('Fold_%d_filtered_g-mean' % current_fold)
				for filtered_gmean in filtered_gmeans:
					sys.stdout.write('\t%.4f%%' % filtered_gmean)
					filtered_perfomance_file.write('\t%.4f' % filtered_gmean)
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')

				sys.stdout.write('Fold %d filtered macro g-mean:' % current_fold)
				filtered_perfomance_file.write('Fold_%d_filtered_macro_g-mean' % current_fold)
				for filtered_macro_gmean in filtered_macro_gmeans:
					sys.stdout.write('\t%.4f%%' % filtered_macro_gmean)
					filtered_perfomance_file.write('\t%.4f' % filtered_macro_gmean)
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')

				sys.stdout.write('Fold %d filtered classified ratio:' % current_fold)
				filtered_perfomance_file.write('Fold_%d_filtered_classified_ratio' % current_fold)
				for filtered_classified_ratio in filtered_classified_ratios:
					sys.stdout.write('\t%.4f%%' % filtered_classified_ratio)
					filtered_perfomance_file.write('\t%.4f' % filtered_classified_ratio)
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')

				try:
					self.extract_per_epoch_times(current_fold, time_epochs_filename)
				except:
					try:
						self.write_SAE_training_times(current_fold, time_sdae_filename)
					except:
						self.write_classifier_training_times(current_fold, time_RF_filename)

				self.write_test_times(current_fold, test_time_filename)

				self.write_predictions(self.categorical_labels[test], predictions, dl_model, current_fold)
				self.write_filtered_predictions(filtered_categorical_labels_test_list, filtered_predictions_list, dl_model, current_fold)

			self.debug_log.info('stratified_cross_validation: Ending fold %d execution' % current_fold)
			current_fold += 1

		# Evaluation of macro performance metrics
		mean_accuracy = numpy.mean(cvscores_pred)
		std_accuracy = numpy.std(cvscores_pred)

		mean_fmeasure = numpy.mean(fscores_pred)
		std_fmeasure = numpy.std(fscores_pred)

		mean_gmean = numpy.mean(gmeans_pred)
		std_gmean = numpy.std(gmeans_pred)

		mean_macro_gmean = numpy.mean(macro_gmeans_pred)
		std_macro_gmean = numpy.std(macro_gmeans_pred)

		print("Macro accuracy: %.4f%% (+/- %.4f%%)" % (mean_accuracy, std_accuracy))
		print("Macro F-measure: %.4f%% (+/- %.4f%%)" % (mean_fmeasure, std_fmeasure))
		print("Average g-mean: %.4f%% (+/- %.4f%%)" % (mean_gmean, std_gmean))
		print("Average macro g-mean: %.4f%% (+/- %.4f%%)" % (mean_macro_gmean, std_macro_gmean))

		with open('%s/performance/%s_performance.dat' % (self.outdir, dl_model.__name__), 'a') as performance_file:
			performance_file.write('Macro_total\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\n' % \
			(mean_accuracy, std_accuracy, mean_fmeasure, std_fmeasure, mean_gmean, std_gmean, mean_macro_gmean, std_macro_gmean))

		# Evaluation of macro performance metrics on training set
		training_mean_accuracy = numpy.mean(training_cvscores_pred)
		training_std_accuracy = numpy.std(training_cvscores_pred)

		training_mean_fmeasure = numpy.mean(training_fscores_pred)
		training_std_fmeasure = numpy.std(training_fscores_pred)

		training_mean_gmean = numpy.mean(training_gmeans_pred)
		training_std_gmean = numpy.std(training_gmeans_pred)

		training_mean_macro_gmean = numpy.mean(training_macro_gmeans_pred)
		training_std_macro_gmean = numpy.std(training_macro_gmeans_pred)

		print("Training macro accuracy: %.4f%% (+/- %.4f%%)" % (training_mean_accuracy, training_std_accuracy))
		print("Training macro F-measure: %.4f%% (+/- %.4f%%)" % (training_mean_fmeasure, training_std_fmeasure))
		print("Training average g-mean: %.4f%% (+/- %.4f%%)" % (training_mean_gmean, training_std_gmean))
		print("Training average macro g-mean: %.4f%% (+/- %.4f%%)" % (training_mean_macro_gmean, training_std_macro_gmean))

		with open('%s/performance/%s_training_performance.dat' % (self.outdir, dl_model.__name__), 'a') as training_performance_file:
			training_performance_file.write('Macro_total\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\n' % \
			(training_mean_accuracy, training_std_accuracy, training_mean_fmeasure, training_std_fmeasure, training_mean_gmean, training_std_gmean, training_mean_macro_gmean, training_std_macro_gmean))

		# Average confusion matrix
		average_cnf_matrix = self.compute_average_confusion_matrix(norm_cms_pred, num_fold)
		average_cnf_matrix_filename = '%s/confusion_matrix/%s_cnf_matrix.dat' % (self.outdir, dl_model.__name__)
		sys.stdout.write('Macro_total_confusion_matrix:\n')
		self.write_average_cnf_matrix(average_cnf_matrix, average_cnf_matrix_filename)

		# Macro top-K accuracy
		mean_topk_accuracies = []
		std_topk_accuracies = []

		topk_accuracies_pred_trans = list(map(list, zip(*topk_accuracies_pred)))
		for cvtopk_accuracies in topk_accuracies_pred_trans:
			mean_topk_accuracies.append(numpy.mean(cvtopk_accuracies))
			std_topk_accuracies.append(numpy.std(cvtopk_accuracies))

		with open('%s/performance/%s_top-k_accuracy.dat' % (self.outdir, dl_model.__name__), 'a') as topk_accuracy_file:
			topk_accuracy_file.write('Macro_total')
			for i, mean_topk_accuracy in enumerate(mean_topk_accuracies):
				print("Macro top-%d accuracy: %.4f%% (+/- %.4f%%)" % (i, mean_topk_accuracy, std_topk_accuracies[i]))
				topk_accuracy_file.write('\t%.4f\t%.4f' % (mean_topk_accuracy, std_topk_accuracies[i]))
			topk_accuracy_file.write('\n')

		# Macro accuracy, F-measure, and g-mean with reject option
		mean_filtered_accuracies = []
		std_filtered_accuracies = []

		mean_filtered_fmeasures = []
		std_filtered_fmeasures = []

		mean_filtered_gmeans = []
		std_filtered_gmeans = []

		mean_filtered_macro_gmeans = []
		std_filtered_macro_gmeans = []

		mean_filtered_classified_ratios = []
		std_filtered_classified_ratios = []

		filtered_cvscores_pred_trans = list(map(list, zip(*filtered_cvscores_pred)))
		filtered_fscores_pred_trans = list(map(list, zip(*filtered_fscores_pred)))
		filtered_gmeans_pred_trans = list(map(list, zip(*filtered_gmeans_pred)))
		filtered_macro_gmeans_pred_trans = list(map(list, zip(*filtered_macro_gmeans_pred)))
		filtered_classified_ratios_pred_trans = list(map(list, zip(*filtered_classified_ratios_pred)))
		for i, filtered_cvscores in enumerate(filtered_cvscores_pred_trans):
			mean_filtered_accuracies.append(numpy.mean(filtered_cvscores))
			std_filtered_accuracies.append(numpy.std(filtered_cvscores))
			mean_filtered_fmeasures.append(numpy.mean(filtered_fscores_pred_trans[i]))
			std_filtered_fmeasures.append(numpy.std(filtered_fscores_pred_trans[i]))
			mean_filtered_gmeans.append(numpy.mean(filtered_gmeans_pred_trans[i]))
			std_filtered_gmeans.append(numpy.std(filtered_gmeans_pred_trans[i]))
			mean_filtered_macro_gmeans.append(numpy.mean(filtered_macro_gmeans_pred_trans[i]))
			std_filtered_macro_gmeans.append(numpy.std(filtered_macro_gmeans_pred_trans[i]))
			mean_filtered_classified_ratios.append(numpy.mean(filtered_classified_ratios_pred_trans[i]))
			std_filtered_classified_ratios.append(numpy.std(filtered_classified_ratios_pred_trans[i]))

		with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
			sys.stdout.write('Macro_total_filtered_accuracy:')
			filtered_perfomance_file.write('Macro_total_filtered_accuracy')
			for i, mean_filtered_accuracy in enumerate(mean_filtered_accuracies):
				sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_accuracy, std_filtered_accuracies[i]))
				filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_accuracy, std_filtered_accuracies[i]))
			sys.stdout.write('\n')
			filtered_perfomance_file.write('\n')

		with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
			sys.stdout.write('Macro_total_filtered_F-measure:')
			filtered_perfomance_file.write('Macro_total_filtered_F-measure')
			for i, mean_filtered_fmeasure in enumerate(mean_filtered_fmeasures):
				sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_fmeasure, std_filtered_fmeasures[i]))
				filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_fmeasure, std_filtered_fmeasures[i]))
			sys.stdout.write('\n')
			filtered_perfomance_file.write('\n')

		with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
			sys.stdout.write('Average_total_filtered_g-mean:')
			filtered_perfomance_file.write('Average_total_filtered_g-mean')
			for i, mean_filtered_gmean in enumerate(mean_filtered_gmeans):
				sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_gmean, std_filtered_gmeans[i]))
				filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_gmean, std_filtered_gmeans[i]))
			sys.stdout.write('\n')
			filtered_perfomance_file.write('\n')

		with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
			sys.stdout.write('Average_total_filtered_macro_g-mean:')
			filtered_perfomance_file.write('Average_total_filtered_macro_g-mean')
			for i, mean_filtered_macro_gmean in enumerate(mean_filtered_macro_gmeans):
				sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_macro_gmean, std_filtered_macro_gmeans[i]))
				filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_macro_gmean, std_filtered_macro_gmeans[i]))
			sys.stdout.write('\n')
			filtered_perfomance_file.write('\n')

		with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
			sys.stdout.write('Macro_total_filtered_classified_ratio:')
			filtered_perfomance_file.write('Macro_total_filtered_classified_ratio')
			for i, mean_filtered_classified_ratio in enumerate(mean_filtered_classified_ratios):
				sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_classified_ratio, std_filtered_classified_ratios[i]))
				filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_classified_ratio, std_filtered_classified_ratios[i]))
			sys.stdout.write('\n')
			filtered_perfomance_file.write('\n')

		# Average filtered confusion matrix
		filtered_norm_cms_pred_trans = list(map(list, zip(*filtered_norm_cms_pred)))
		for i, filtered_norm_cms in enumerate(filtered_norm_cms_pred_trans):
			try:
				filtered_average_cnf_matrix = self.compute_average_confusion_matrix(filtered_norm_cms, num_fold)
				filtered_average_cnf_matrix_filename = '%s/confusion_matrix/%s_filtered_cnf_matrix_%s.dat' % (self.outdir, dl_model.__name__, str(round(self.gamma_range[i], 1)).replace('.', '_'))
				sys.stdout.write('Macro_total_filtered_confusion_matrix_%s:\n' % str(round(self.gamma_range[i], 1)).replace('.', '_'))
				self.write_average_cnf_matrix(filtered_average_cnf_matrix, filtered_average_cnf_matrix_filename)
			except (ValueError, IndexError) as exc:
				sys.stderr.write('%s\n' % exc)
				sys.stderr.write('Error in computing macro total filtered confusion_matrix for gamma %s.\n' % str(round(self.gamma_range[i], 1)).replace('.', '_'))

		self.debug_log.info('Ending execution of stratified cross validation')


	def train_model(self, model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes):
		'''
		Trains (fit) the keras model given as input
		'''

		one_hot_categorical_labels_train = keras.utils.to_categorical(categorical_labels_train, num_classes=num_classes)
		model.fit(samples_train, one_hot_categorical_labels_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)


	def train_SAE(self, model, samples_train, categorical_labels_train, num_classes):
		'''
		Trains (fit) the SAE model given as input
		'''

		# Unsupervised layer-wise pre-training
		pretrain_time_begin = time.time()
		model.pretrain(samples_train)
		pretrain_time_end = time.time()
		self.pretrain_time = pretrain_time_end - pretrain_time_begin

		self.debug_log.info('SAE: Unsupervised layer-wise pre-training completed')
		print('Unsupervised layer-wise pre-training completed')

		# Supervised fine tuning
		one_hot_categorical_labels_train = keras.utils.to_categorical(categorical_labels_train, num_classes=self.num_classes)
		train_time_begin = time.time()
		model.fit(samples_train, one_hot_categorical_labels_train)
		train_time_end = time.time()
		self.train_time = train_time_end - train_time_begin

		self.debug_log.info('SAE: Supervised training phase completed')
		print('Supervised training phase completed')


	def train_multimodal_model(self, multimodal_model, epochs, batch_size, callbacks, samples_train_list, categorical_labels_train, num_classes, num_samples_c0, num_samples_c1):
		'''
		Trains (fit) the keras multimodal model given as input
		'''

		one_hot_categorical_labels_train = keras.utils.to_categorical(categorical_labels_train, num_classes=num_classes)

		class_imbalance = {
			0: (num_samples_c0 + num_samples_c1) / num_samples_c0,
			1: (num_samples_c0 + num_samples_c1) / num_samples_c1
		}

		multimodal_model.fit(x = samples_train_list, y = one_hot_categorical_labels_train, class_weight = class_imbalance, \
		epochs = epochs, batch_size = batch_size, callbacks = callbacks, verbose=2)


	def test_model(self, model, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Test the keras model given as input with predict_classes()
		'''

		predictions = model.predict_classes(samples_test, verbose=2)

		# Calculate soft predictions and test time
		test_time_begin = time.time()
		soft_values = model.predict(samples_test, verbose=2)
		test_time_end = time.time()
		self.test_time = test_time_end - test_time_begin

		training_predictions = model.predict_classes(samples_train, verbose=2)

		# print(len(categorical_labels_test))
		# print(categorical_labels_test)

		# print(len(predictions))
		# print(predictions)

		# print(len(soft_values))
		# print(soft_values)

		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))

		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(categorical_labels_train, training_predictions)
		training_fmeasure = sklearn.metrics.f1_score(categorical_labels_train, training_predictions, average='macro')
		training_gmean = self.compute_g_mean(categorical_labels_train, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_train, training_predictions, average=None))

		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)

		# Top-K accuracy
		topk_accuracies = []
		for k in self.K:
			topk_accuracy = self.compute_topk_accuracy(soft_values, categorical_labels_test, k)
			topk_accuracies.append(topk_accuracy)

		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		for gamma in self.gamma_range:
			categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio = \
			self.compute_filtered_performance(model, samples_test, soft_values, categorical_labels_test, gamma)
			filtered_categorical_labels_test_list.append(categorical_labels_test_filtered)
			filtered_predictions_list.append(filtered_predictions)
			filtered_accuracies.append(filtered_accuracy)
			filtered_fmeasures.append(filtered_fmeasure)
			filtered_gmeans.append(filtered_gmean)
			filtered_macro_gmeans.append(filtered_macro_gmean)
			filtered_norm_cnf_matrices.append(filtered_norm_cnf_matrix)
			filtered_classified_ratios.append(filtered_classified_ratio)

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, filtered_norm_cnf_matrices, filtered_classified_ratios


	def test_SAE(self, model, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Test the SAE model given as input with predict_classes()
		'''

		# Calculate logit predictions and test time
		test_time_begin = time.time()
		logit_predictions = model.predict(samples_test)						# Predictions are logit values
		test_time_end = time.time()
		self.test_time = test_time_end - test_time_begin

		categorical_predictions = logit_predictions.argmax(1)

		if self.dataset == 1:
			soft_values = numpy.asarray(self.convert_logit_predictions_FB_FBM(logit_predictions))
		else:
			soft_values = numpy.asarray(self.convert_logit_predictions_multi(logit_predictions))

		training_logit_predictions = model.predict(samples_train)
		training_categorical_predictions = training_logit_predictions.argmax(1)

		# print(len(categorical_labels_test))
		# print(categorical_labels_test)

		# print(len(categorical_predictions))
		# print(categorical_predictions)

		# print(len(logit_predictions))
		# print(logit_predictions)

		# print(len(soft_values))
		# print(soft_values)

		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, categorical_predictions)
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, categorical_predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, categorical_predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, categorical_predictions, average=None))

		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(categorical_labels_train, training_categorical_predictions)
		training_fmeasure = sklearn.metrics.f1_score(categorical_labels_train, training_categorical_predictions, average='macro')
		training_gmean = self.compute_g_mean(categorical_labels_train, training_categorical_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_train, training_categorical_predictions, average=None))

		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, categorical_predictions)

		# Top-K accuracy
		topk_accuracies = []
		for k in self.K:
			try:
				topk_accuracy = self.compute_topk_accuracy(soft_values, categorical_labels_test, k)
				topk_accuracies.append(topk_accuracy)
			except Exception as exc:
				sys.stderr.write('%s\n' % exc)

		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		for gamma in self.gamma_range:
			categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio = \
			self.compute_filtered_performance(model, samples_test, soft_values, categorical_labels_test, gamma)
			filtered_categorical_labels_test_list.append(categorical_labels_test_filtered)
			filtered_predictions_list.append(filtered_predictions)
			filtered_accuracies.append(filtered_accuracy)
			filtered_fmeasures.append(filtered_fmeasure)
			filtered_gmeans.append(filtered_gmean)
			filtered_macro_gmeans.append(filtered_macro_gmean)
			filtered_norm_cnf_matrices.append(filtered_norm_cnf_matrix)
			filtered_classified_ratios.append(filtered_classified_ratio)

		return categorical_predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, filtered_norm_cnf_matrices, filtered_classified_ratios


	def test_classifier(self, clf, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Test the sklearn classifier given as input with predict_classes()
		'''

		predictions = clf.predict(samples_test)

		# Calculate soft predictions and test time
		test_time_begin = time.time()
		soft_values = clf.predict_proba(samples_test)
		test_time_end = time.time()
		self.test_time = test_time_end - test_time_begin

		training_predictions = clf.predict(samples_train)

		# print(len(categorical_labels_test))
		# print(categorical_labels_test)

		# print(len(predictions))
		# print(predictions)

		# print(len(soft_values))
		# print(soft_values)

		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))

		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(categorical_labels_train, training_predictions)
		training_fmeasure = sklearn.metrics.f1_score(categorical_labels_train, training_predictions, average='macro')
		training_gmean = self.compute_g_mean(categorical_labels_train, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_train, training_predictions, average=None))

		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)

		# Top-K accuracy
		topk_accuracies = []
		for k in self.K:
			try:
				topk_accuracy = self.compute_topk_accuracy(soft_values, categorical_labels_test, k)
				topk_accuracies.append(topk_accuracy)
			except Exception as exc:
				sys.stderr.write('%s\n' % exc)

		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		for gamma in self.gamma_range:
			categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio = \
			self.compute_classifier_filtered_performance(clf, samples_test, soft_values, categorical_labels_test, gamma)
			filtered_categorical_labels_test_list.append(categorical_labels_test_filtered)
			filtered_predictions_list.append(filtered_predictions)
			filtered_accuracies.append(filtered_accuracy)
			filtered_fmeasures.append(filtered_fmeasure)
			filtered_gmeans.append(filtered_gmean)
			filtered_macro_gmeans.append(filtered_macro_gmean)
			filtered_norm_cnf_matrices.append(filtered_norm_cnf_matrix)
			filtered_classified_ratios.append(filtered_classified_ratio)

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, filtered_norm_cnf_matrices, filtered_classified_ratios


	def test_multimodal_model(self, multimodal_model, samples_train_list, categorical_labels_train, samples_test_list, categorical_labels_test):
		'''
		Test the keras multimodal model given as input with predict()
		'''

		# Calculate soft predictions and test time
		test_time_begin = time.time()
		soft_values = multimodal_model.predict(samples_test_list, verbose=2)
		test_time_end = time.time()
		self.test_time = test_time_end - test_time_begin

		predictions = soft_values.argmax(axis=-1)

		training_soft_values = multimodal_model.predict(samples_train_list, verbose=2)
		training_predictions = training_soft_values.argmax(axis=-1)

		# print(len(categorical_labels_test))
		# print(categorical_labels_test)

		# print(len(predictions))
		# print(predictions)

		# print(len(soft_values))
		# print(soft_values)

		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))

		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(categorical_labels_train, training_predictions)
		training_fmeasure = sklearn.metrics.f1_score(categorical_labels_train, training_predictions, average='macro')
		training_gmean = self.compute_g_mean(categorical_labels_train, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_train, training_predictions, average=None))

		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)

		# Top-K accuracy
		topk_accuracies = []
		for k in self.K:
			topk_accuracy = self.compute_topk_accuracy(soft_values, categorical_labels_test, k)
			topk_accuracies.append(topk_accuracy)

		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		for gamma in self.gamma_range:
			categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio = \
			self.compute_multimodal_filtered_performance(multimodal_model, samples_test_list, soft_values, categorical_labels_test, gamma)
			filtered_categorical_labels_test_list.append(categorical_labels_test_filtered)
			filtered_predictions_list.append(filtered_predictions)
			filtered_accuracies.append(filtered_accuracy)
			filtered_fmeasures.append(filtered_fmeasure)
			filtered_gmeans.append(filtered_gmean)
			filtered_macro_gmeans.append(filtered_macro_gmean)
			filtered_norm_cnf_matrices.append(filtered_norm_cnf_matrix)
			filtered_classified_ratios.append(filtered_classified_ratio)

		return predictions, accuracy, fmeasure, gmean, macro_gmean, training_accuracy, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, filtered_norm_cnf_matrices, filtered_classified_ratios


	def compute_norm_confusion_matrix(self, y_true, y_pred):
		'''
		Compute normalized confusion matrix
		'''

		sorted_labels = sorted(set(self.categorical_labels))

		cnf_matrix = confusion_matrix(y_true, y_pred, labels=sorted_labels)
		norm_cnf_matrix = preprocessing.normalize(cnf_matrix, axis=1, norm='l1')

		return norm_cnf_matrix


	def compute_average_confusion_matrix_old(self, norm_cms_pred, num_fold):
		'''
		Compute average confusion matrix over num_fold
		'''

		# Average confusion matrix
		average_cnf_matrix = numpy.zeros((self.num_classes, self.num_classes), dtype=float)

		for norm_cnf_matrix in norm_cms_pred:
			average_cnf_matrix += norm_cnf_matrix
		average_cnf_matrix = average_cnf_matrix / num_fold

		return average_cnf_matrix


	def compute_average_confusion_matrix(self, norm_cms_pred, num_fold):
		'''
		Compute average confusion matrix over num_fold
		'''

		average_cnf_matrix = []

		for i in range(self.num_classes):
			average_cnf_matrix_row = numpy.zeros(self.num_classes, dtype=float)
			non_zero_rows = 0

			for norm_cnf_matrix in norm_cms_pred:
				if numpy.any(norm_cnf_matrix[i]):
					average_cnf_matrix_row = numpy.add(average_cnf_matrix_row, norm_cnf_matrix[i])
					non_zero_rows += 1

			average_cnf_matrix_row = average_cnf_matrix_row / non_zero_rows
			average_cnf_matrix.append(average_cnf_matrix_row)

		return average_cnf_matrix


	def compute_g_mean(self, y_true, y_pred):
		'''
		Compute g-mean as the geometric mean of the recalls of all classes
		'''

		recalls = sklearn.metrics.recall_score(y_true, y_pred, average=None)
		nonzero_recalls = recalls[recalls != 0]

		is_zero_recall = False
		unique_y_true = list(set(y_true))

		for i, recall in enumerate(recalls):
			if recall == 0 and i in unique_y_true:
				is_zero_recall = True
				self.debug_log.error('compute_g_mean: zero-recall obtained, class %s has no sample correcly classified.' % self.label_class[i])

		if is_zero_recall:
			gmean = scipy.stats.mstats.gmean(recalls)
		else:
			gmean = scipy.stats.mstats.gmean(nonzero_recalls)

		return gmean


	def compute_topk_accuracy(self, soft_values, categorical_labels_test, k):
		'''
		Compute top-k accuracy for a given value of k
		'''

		predictions_indices = numpy.argsort(-soft_values, 1)
		predictions_topk_indices = predictions_indices[:,0:k]

		accuracies = numpy.zeros(categorical_labels_test.shape)
		for i in range(k):
			accuracies = accuracies + numpy.equal(predictions_topk_indices[:,i], categorical_labels_test)

		topk_accuracy = sum(accuracies) / categorical_labels_test.size

		return topk_accuracy


	def compute_filtered_performance(self, model, samples_test, soft_values, categorical_labels_test, gamma):
		'''
		Compute filtered performance for a given value of gamma
		'''

		pred_indices = numpy.greater(soft_values.max(1), gamma)
		num_rej_samples = categorical_labels_test.shape - numpy.sum(pred_indices)		# Number of unclassified samples

		samples_test_filtered = samples_test[numpy.nonzero(pred_indices)]
		categorical_labels_test_filtered = categorical_labels_test[numpy.nonzero(pred_indices)]

		try:
			filtered_predictions = model.predict_classes(samples_test_filtered, verbose=2)
		except AttributeError:
			filtered_logit_predictions = model.predict(samples_test_filtered)		# Predictions are logit values
			try:
				filtered_predictions = filtered_logit_predictions.argmax(1)
			except AttributeError:
				self.debug_log.error('Empty list returned by model.predict()')
				sys.stderr.write('Empty list returned by model.predict()\n')
				return numpy.nan, [], numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.array([[numpy.nan] * self.num_classes, [numpy.nan] * self.num_classes]), numpy.nan

		filtered_accuracy = sklearn.metrics.accuracy_score(categorical_labels_test_filtered, filtered_predictions)
		filtered_fmeasure = sklearn.metrics.f1_score(categorical_labels_test_filtered, filtered_predictions, average='macro')
		filtered_gmean = self.compute_g_mean(categorical_labels_test_filtered, filtered_predictions)
		filtered_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test_filtered, filtered_predictions, average=None))
		filtered_norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test_filtered, filtered_predictions)
		filtered_classified_ratio = numpy.sum(pred_indices) / categorical_labels_test.shape

		return categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio


	def compute_classifier_filtered_performance(self, clf, samples_test, soft_values, categorical_labels_test, gamma):
		'''
		Compute filtered performance of a sklearn classifier for a given value of gamma
		'''

		pred_indices = numpy.greater(soft_values.max(1), gamma)
		num_rej_samples = categorical_labels_test.shape - numpy.sum(pred_indices)		# Number of unclassified samples

		samples_test_filtered = samples_test[numpy.nonzero(pred_indices)]
		categorical_labels_test_filtered = categorical_labels_test[numpy.nonzero(pred_indices)]

		filtered_predictions = clf.predict(samples_test_filtered)

		filtered_accuracy = sklearn.metrics.accuracy_score(categorical_labels_test_filtered, filtered_predictions)
		filtered_fmeasure = sklearn.metrics.f1_score(categorical_labels_test_filtered, filtered_predictions, average='macro')
		filtered_gmean = self.compute_g_mean(categorical_labels_test_filtered, filtered_predictions)
		filtered_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test_filtered, filtered_predictions, average=None))
		filtered_norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test_filtered, filtered_predictions)
		filtered_classified_ratio = numpy.sum(pred_indices) / categorical_labels_test.shape

		return categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio


	def compute_multimodal_filtered_performance(self, multimodal_model, samples_test_list, soft_values, categorical_labels_test, gamma):
		'''
		Compute multimodal filtered performance for a given value of gamma
		'''

		pred_indices = numpy.greater(soft_values.max(1), gamma)
		num_rej_samples = categorical_labels_test.shape - numpy.sum(pred_indices)		# Number of unclassified samples

		samples_test_filtered_list = []
		for samples_test in samples_test_list:
			samples_test_filtered_list.append(samples_test[numpy.nonzero(pred_indices)])
		categorical_labels_test_filtered = categorical_labels_test[numpy.nonzero(pred_indices)]

		filtered_soft_values = multimodal_model.predict(samples_test_filtered_list)
		filtered_predictions = filtered_soft_values.argmax(axis=-1)

		filtered_accuracy = sklearn.metrics.accuracy_score(categorical_labels_test_filtered, filtered_predictions)
		filtered_fmeasure = sklearn.metrics.f1_score(categorical_labels_test_filtered, filtered_predictions, average='macro')
		filtered_gmean = self.compute_g_mean(categorical_labels_test_filtered, filtered_predictions)
		filtered_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test_filtered, filtered_predictions, average=None))
		filtered_norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test_filtered, filtered_predictions)
		filtered_classified_ratio = numpy.sum(pred_indices) / categorical_labels_test.shape

		return categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio


	def extract_per_epoch_times(self, current_fold, time_epochs_filename):
		'''
		Extract and print per-epoch times
		'''

		epoch_times = self.time_callback.times
		with open(time_epochs_filename, 'a') as time_epochs_file:
			time_epochs_file.write('Fold_%d\t' % current_fold)
			for epoch_time in epoch_times:
				time_epochs_file.write('\t%.4f' % epoch_time)
			time_epochs_file.write('\n')


	def convert_logit_predictions_FB_FBM(self, logit_predictions):
		'''
		Convert logit predictions to soft values for FB-FBM (binary) dataset
		'''

		numpy.seterr(all='raise')

		soft_values = []
		for sample_logit_predictions in logit_predictions:
			try:
				sample_logit_predictions = numpy.float_(sample_logit_predictions)
				# print(sample_logit_predictions)
				# print(type(sample_logit_predictions))

				b = max(sample_logit_predictions)
				# print(b)
				# print(type(b))

				sample_predictions = [ numpy.exp(numpy.float64(sample_logit_prediction - b)) for sample_logit_prediction in sample_logit_predictions ]
				sample_soft_values = [ sample_prediction / sum(sample_predictions) for sample_prediction in sample_predictions ]
				soft_values.append(sample_soft_values)
				# print(sample_soft_values)
				# print(type(sample_soft_values))

			except FloatingPointError as exc:
				sys.stderr.write('%s\n' % exc)

				if sample_logit_predictions[0] > sample_logit_predictions[1]:
					soft_values.append([1.0, 0.0])
				else:
					soft_values.append([0.0, 1.0])

		numpy.seterr(all='warn')

		return soft_values


	def convert_logit_predictions_multi(self, logit_predictions):
		'''
		Convert logit predictions to soft values for multi-class dataset
		'''

		numpy.seterr(all='raise')

		soft_values = []
		for sample_logit_predictions in logit_predictions:
			sample_logit_predictions = numpy.float_(sample_logit_predictions)
			# print(sample_logit_predictions)
			# print(type(sample_logit_predictions))

			b = max(sample_logit_predictions)
			# print(b)
			# print(type(b))

			sample_predictions = []
			for sample_logit_prediction in sample_logit_predictions:
				try:
					exp_sample_logit_prediction = numpy.exp(numpy.float64(sample_logit_prediction - b))
				except FloatingPointError as exc:
					sys.stderr.write('%s: %s\n' % (exc, (sample_logit_prediction - b)))
					exp_sample_logit_prediction = 0.0
				sample_predictions.append(exp_sample_logit_prediction)

			sample_soft_values = [ sample_prediction / sum(sample_predictions) for sample_prediction in sample_predictions ]
			soft_values.append(sample_soft_values)
			# print(sample_soft_values)
			# print(type(sample_soft_values))

		numpy.seterr(all='warn')

		return soft_values


	def write_average_cnf_matrix(self, average_cnf_matrix, average_cnf_matrix_filename):
		'''
		Write average confusion matrix (potentially filtered)
		'''

		with open(average_cnf_matrix_filename, 'w') as average_cnf_matrix_file:
			sorted_labels = sorted(set(self.categorical_labels))
			for label in sorted_labels:
				average_cnf_matrix_file.write('%s,' % label)
				sys.stdout.write('%s,' % label)
			average_cnf_matrix_file.write(',\n')
			sys.stdout.write(',\n')

			for i, row in enumerate(average_cnf_matrix):
				for occurences in row:
					average_cnf_matrix_file.write('%s,' % occurences)
					sys.stdout.write('%s,' % occurences)
				average_cnf_matrix_file.write('%s,%s\n' % (sorted_labels[i], self.label_class[int(sorted_labels[i])]))
				sys.stdout.write('%s,%s\n' % (sorted_labels[i], self.label_class[int(sorted_labels[i])]))


	def write_SAE_training_times(self, current_fold, time_sdae_filename):
		'''
		Write SAE pre-training and training times
		'''

		print('Pre-training time: %.4f' % self.pretrain_time)
		print('Training time: %.4f' % self.train_time)

		with open(time_sdae_filename, 'a') as time_sdae_file:
			time_sdae_file.write('Fold_%d\t%.4f\t%.4f\n' % (current_fold, self.pretrain_time, self.train_time))


	def write_classifier_training_times(self, current_fold, time_classifier_filename):
		'''
		Write sklearn classifier training times
		'''

		print('Training time: %.4f' % self.train_time)

		with open(time_classifier_filename, 'a') as time_classifier_file:
			time_classifier_file.write('Fold_%d\t%.4f\n' % (current_fold, self.train_time))


	def write_test_times(self, current_fold, test_time_filename):
		'''
		Write test times
		'''

		print('Test time: %.4f' % self.test_time)

		with open(test_time_filename, 'a') as test_time_file:
			test_time_file.write('Fold_%d\t%.4f\n' % (current_fold, self.test_time))


	def write_predictions(self, categorical_labels_test, predictions, dl_model, current_fold):
		'''
		Write (multimodal) model / classifier predictions
		'''

		try:
			os.makedirs('%s/predictions/fold_%d' % (self.outdir, current_fold))
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				traceback.print_exc(file=sys.stderr)

		predictions_filename = '%s/predictions/fold_%d/%s_fold_%d_predictions.dat' % (self.outdir, current_fold, dl_model.__name__, current_fold)
		with open(predictions_filename, 'w') as predictions_file:
			predictions_file.write('Actual\t%s\n' % dl_model.__name__)
			for i, prediction in enumerate(predictions):
				predictions_file.write('%s\t%s\n' % (categorical_labels_test[i], prediction))


	def write_filtered_predictions(self, filtered_categorical_labels_test_list, filtered_predictions_list, dl_model, current_fold):
		'''
		Write (multimodal) model / classifier filtered predictions
		'''

		try:
			os.makedirs('%s/predictions/fold_%d' % (self.outdir, current_fold))
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				traceback.print_exc(file=sys.stderr)

		for i, filtered_predictions in enumerate(filtered_predictions_list):
			filtered_predictions_filename = '%s/predictions/fold_%d/%s_fold_%d_filtered_predictions_%s.dat' % \
			(self.outdir, current_fold, dl_model.__name__, current_fold, str(round(self.gamma_range[i], 1)).replace('.', '_'))
			with open(filtered_predictions_filename, 'w') as filtered_predictions_file:
				filtered_predictions_file.write('Actual\t%s\n' % dl_model.__name__)
				for j, filtered_prediction in enumerate(filtered_predictions):
					filtered_predictions_file.write('%s\t%s\n' % (filtered_categorical_labels_test_list[i][j], filtered_prediction))


	def deserialize_dataset(self, include_ports, is_multimodal, view_index):
		'''
		Deserialize TC datasets to extract samples and (categorical) labels to perform classification using one of the state-of-the-art DL (multimodal) approaches
		'''

		self.debug_log.info('Starting datasets deserialization')

		samples_list = []
		categorical_labels_list = []
		for pickle_dataset_filename in self.pickle_dataset_filenames:
			with open(pickle_dataset_filename, 'rb') as pickle_dataset_file:
				samples_list.append(pickle.load(pickle_dataset_file))
				for i in range(self.view):
					categorical_labels_list.append(pickle.load(pickle_dataset_file))
			self.debug_log.debug('Dataset %s deserialized' % pickle_dataset_filename)

		if not is_multimodal:
			self.samples = samples_list[0]
			self.categorical_labels = categorical_labels_list[self.view-1]

			self.num_samples = numpy.shape(self.samples)[0]
			self.input_dim = numpy.shape(self.samples)[1]
			self.num_classes = numpy.max(self.categorical_labels) + 1

			if dl_network == 4 or dl_network == 5 or dl_network == 6 or dl_network == 11:
				if include_ports:
					self.samples = numpy.asarray(self.samples)
				else:
					self.samples = numpy.asarray(self.samples)[:,:,2:6]
				self.input_dim_2 = numpy.shape(self.samples)[2]

			# print(self.samples)
			# print(self.categorical_labels)
			# print(self.num_samples)
			# print(self.input_dim)
			# if dl_network == 4 or dl_network == 5 or dl_network == 6 or dl_network == 11: print(self.input_dim_2)
			# print(self.num_classes)

		else:
			num_samples_list = [numpy.shape(samples)[0] for samples in samples_list]
			num_classes_list = [numpy.max(categorical_labels) for categorical_labels in categorical_labels_list]

			if len(set(num_samples_list)) <= 1 and len(set(num_classes_list)) <= 1:
				self.samples = samples_list[0]					# used to calculate indices for 10-fold validation
				self.categorical_labels = categorical_labels_list[0]
				self.num_samples = num_samples_list[0]
				self.num_classes = num_classes_list[0] + 1

			elif len(set(num_samples_list)) > 1:
				raise DLArchitecturesException('The datasets provided have a different number of samples.')

			else:
				raise DLArchitecturesException('The datasets provided have a different number of classes.')

			self.samples_list = samples_list
			self.categorical_labels_list = categorical_labels_list

			if not numpy.array_equal(self.categorical_labels, self.categorical_labels_list[1]):
				raise DLArchitecturesException('The datasets provided have different labels.')

			self.input_dims_list = []

			for i, pickle_dataset_filename in enumerate(self.pickle_dataset_filenames):
				if 'lopez' in pickle_dataset_filename.lower():
					if include_ports:
						self.samples_list[i] = numpy.asarray(self.samples_list[i])
					else:
						self.samples_list[i] = numpy.asarray(self.samples_list[i])[:,:,2:6]
					self.input_dims_list.append(numpy.shape(self.samples_list[i])[1:])
				else:
					self.input_dims_list.append(numpy.shape(self.samples_list[i])[1])

			# print(self.samples_list)
			# print(self.categorical_labels_list)
			# print(self.num_samples)
			# print(self.input_dims_list)
			# print(self.num_classes)

		self.debug_log.info('Ending datasets deserialization')


if __name__ == "__main__":

	if len(sys.argv) < 7:
		print('usage:', sys.argv[0], '<PICKLE_DATASET_NUMBER>', '<PICKLE_DATASET_1>', '[PICKLE_DATASET_2]', '...', '[PICKLE_DATASET_N]', '<DL_NETWORK_INDEX>', '<DATASET_INDEX>', '<VIEW_INDEX>', '<OUTDIR>')
		sys.exit(1)
	elif len(sys.argv) < (int(sys.argv[1]) + 6):
		print('usage:', sys.argv[0], sys.argv[1], '<PICKLE_DATASET_1>', '...', '<PICKLE_DATASET_%s>' % sys.argv[1], '<DL_NETWORK_INDEX>', '<DATASET_INDEX>', '<VIEW_INDEX>', '<OUTDIR>')
		sys.exit(1)

	try:
		pickle_dataset_number = int(sys.argv[1])
	except ValueError:
		raise DLArchitecturesException('pickle_dataset_number must be an integer.')

	pickle_dataset_filenames = []
	for i in range(pickle_dataset_number):
		pickle_dataset_filenames.append(sys.argv[i + 2])

	is_multimodal = False
	if len(pickle_dataset_filenames) > 1:
		is_multimodal = True

	try:
		dl_network = int(sys.argv[pickle_dataset_number + 2])
	except ValueError:
		dl_network = -1

    # dataset == 1 is FB/FBM; dataset == 2 is Android; dataset == 3 is iOS; dataset == 4 is VPN/nonVPN.
	try:
		dataset = int(sys.argv[pickle_dataset_number + 3])
	except ValueError:
		dataset = -1

	#TODO: Manage VIEW_INDEX to select multi-task view
	try:
		view = int(sys.argv[pickle_dataset_number + 4])
	except ValueError:
		view = -1

	outdir = sys.argv[pickle_dataset_number + 5]

	try:
		os.makedirs('%s/confusion_matrix' % outdir)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			traceback.print_exc(file=sys.stderr)

	try:
		os.makedirs('%s/performance' % outdir)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			traceback.print_exc(file=sys.stderr)

	try:
		os.makedirs('%s/predictions' % outdir)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			traceback.print_exc(file=sys.stderr)

	# Set the number of folds for stratified cross-validation
	num_fold = 10

	# Set random seed for reproducibility
	seed = int(time.time())
	# print(seed)
	numpy.random.seed(seed)

	include_ports = False				# It should be set to False to avoid biased (port-dependent) inputs
	dl_architectures = DLArchitectures(pickle_dataset_filenames, dataset, view, outdir, include_ports, is_multimodal)

	if dl_network == 1:
		dl_architectures.stratified_cross_validation(dl_architectures.wang2017endtoend_1DCNN, num_fold, seed, do_expand_dims = True, do_normalization = False)
	elif dl_network == 2:
		dl_architectures.stratified_cross_validation(dl_architectures.wang2017malware_2DCNN, num_fold, seed, do_expand_dims = True, do_normalization = False)
	elif dl_network == 3:
		dl_architectures.stratified_cross_validation(dl_architectures.lotfollahi2017deep_1DCNN, num_fold, seed, do_expand_dims = True, do_normalization = False)
	elif dl_network == 4:
		dl_architectures.stratified_cross_validation(dl_architectures.lopez2017network_CNN_1, num_fold, seed, do_expand_dims = True, do_normalization = True)
	elif dl_network == 5:
		dl_architectures.stratified_cross_validation(dl_architectures.lopez2017network_RNN_1, num_fold, seed, do_expand_dims = False, do_normalization = True)
	elif dl_network == 6:
		dl_architectures.stratified_cross_validation(dl_architectures.lopez2017network_CNN_RNN_2a, num_fold, seed, do_expand_dims = True, do_normalization = True)
	elif dl_network == 7:
		dl_architectures.stratified_cross_validation(dl_architectures.wang2015applications_SAE, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 8:
		dl_architectures.stratified_cross_validation(dl_architectures.lotfollahi2017deep_SAE, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 9:
		dl_architectures.stratified_cross_validation(dl_architectures.single_layer_perceptron, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 10:
		dl_architectures.stratified_cross_validation(dl_architectures.taylor2016appscanner_RF, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 11:
		dl_architectures.stratified_cross_validation(dl_architectures.single_layer_perceptron_lopez, num_fold, seed, do_expand_dims = False, do_normalization = True)
	elif dl_network == 12:
		dl_architectures.stratified_cross_validation(dl_architectures.oh2017traffic_2DCNN, num_fold, seed, do_expand_dims = True, do_normalization = False)
	elif dl_network == 13:
		dl_architectures.stratified_cross_validation(dl_architectures.oh2017traffic_MLP, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 14:
		dl_architectures.stratified_cross_validation(dl_architectures.multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU, num_fold, seed, is_multimodal = is_multimodal)
	elif dl_network == 15:
		dl_architectures.stratified_cross_validation(dl_architectures.multi_layer_perceptron_test_lotfollahi2017deep_SAE, num_fold, seed, do_expand_dims = False, do_normalization = False)
	else:
		sys.stderr.write("Please, provide a valid DL network index in the range 1-15.\n")
