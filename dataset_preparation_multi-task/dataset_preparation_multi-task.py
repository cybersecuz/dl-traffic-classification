#    This script performs a series of pre-processing operations on the TC dataset.
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
Created on 9 nov 2017

@author: antonio
'''

import sys, os, time

import keras, sklearn, numpy
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

import pickle
import logging
from logging.handlers import RotatingFileHandler
from pprint import pprint
import csv


class DatasetPreparationException(Exception):
    '''
    Exception for DatasetPreparation class
    '''
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return str(self.message)


class DatasetPreparation(object):
	'''
	Perform a series of pre-processing operations on the TC dataset and serialize the latter dataset in .pickle format:
		- load the data from a .csv file
		- normalize the dataset in [0, 1]
		- convert string labels to categorical values
	'''
	
	def __init__(self, csv_dataset, input_dim, outdir):
		'''
		Constructor for the class DatasetPreparation.
		Input:
		- csv_dataset (string): TC dataset in .csv format
		- input_dim (int): dimension of the input samples of TC dataset
		- outdir (string): outdir to contain pickle dataset
		'''
		
		if os.path.isfile(csv_dataset):
			self.csv_dataset = csv_dataset
		else:
			err_str = 'The CSV dataset does not exist. Please, provide a valid CSV dataset.'
			raise DatasetPreparationException(err_str)
		
		self.input_dim = int(input_dim)
		self.outdir = outdir
		self.dataset_basename = os.path.splitext(os.path.basename(self.csv_dataset))[0]
		
		self.debug_log = self.setup_logger('debug', '%s/dataset_preparation.log' % self.outdir, logging.DEBUG)
	
	
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
	
	
	def prepare_dataset(self):
		'''
		Perform the following pre-processing operations on the TC dataset:
		- load the data from a .csv file
		- normalize the dataset in [0, 1]
		- convert string labels to categorical values
		'''
		
		self.debug_log.info('Starting dataset preparation')
		
		# Load our TC dataset (note that range() does not consider the last element)
		self.samples = numpy.loadtxt(self.csv_dataset, delimiter=",", usecols = range(0, self.input_dim), dtype='float')
		self.vpn_labels = numpy.loadtxt(self.csv_dataset, delimiter=",", usecols = (self.input_dim,), dtype='str')
		self.class_labels = numpy.loadtxt(self.csv_dataset, delimiter=",", usecols = (self.input_dim+1,), dtype='str')
		self.app_labels = numpy.loadtxt(self.csv_dataset, delimiter=",", usecols = (self.input_dim+2,), dtype='str')
		self.debug_log.info('Dataset loaded')
		
		# Normalize the dataset in [0, 1]
		self.normalized_samples = self.samples / numpy.max(self.samples)
		self.debug_log.info('Dataset normalized')
		
		# Convert the labels from a list of strings to categorical values
		label_encoder = preprocessing.LabelEncoder()
		
		label_vpn_encoder = preprocessing.LabelEncoder()
		label_class_encoder = preprocessing.LabelEncoder()
		label_app_encoder = preprocessing.LabelEncoder()

		label_vpn_encoder.fit(self.vpn_labels)
		label_class_encoder.fit(self.class_labels)
		label_app_encoder.fit(self.app_labels)

		self.categorical_vpn_labels = label_vpn_encoder.transform(self.vpn_labels)
		self.categorical_class_labels = label_class_encoder.transform(self.class_labels)
		self.categorical_app_labels = label_app_encoder.transform(self.app_labels)
		
		# print (self.categorical_vpn_labels)
		# print (self.categorical_class_labels)
		# print (self.categorical_app_labels)

		self.debug_log.info('Labels converted to categorical values')
		
		self.debug_log.info('Ending dataset preparation')


	
	def write_trace_csv_file(self):
			'''
			Saves the first M bytes of the biflows of each capture in CSV format and in the global_csv_filename
			'''


			csv_filename = '%s/%s_categorical_biflows.csv' % (self.outdir, self.dataset_basename)


			os.remove(csv_filename) if os.path.exists(csv_filename) else None


			with open(csv_filename, 'w') as csv_file:

						writer = csv.writer(csv_file, delimiter='\t')

						writer.writerows(zip (self.categorical_vpn_labels,self.vpn_labels,self.categorical_class_labels,self.class_labels, self.categorical_app_labels,self.app_labels))



	def serialize_dataset(self):
		'''
		Serialize TC dataset to later perform classification using DL networks
		'''
		
		self.debug_log.info('Starting dataset serialization')
		
		dataset_filename = '%s/%s_categorical.pickle' % (self.outdir, self.dataset_basename)
		
		if os.path.isfile(dataset_filename):
			os.remove(dataset_filename)
		
		with open(dataset_filename, 'wb') as dataset_file:
			pickle.dump(self.normalized_samples, dataset_file, pickle.HIGHEST_PROTOCOL)
			pickle.dump(self.categorical_vpn_labels, dataset_file, pickle.HIGHEST_PROTOCOL)
			pickle.dump(self.categorical_class_labels, dataset_file, pickle.HIGHEST_PROTOCOL)
			pickle.dump(self.categorical_app_labels, dataset_file, pickle.HIGHEST_PROTOCOL)

		
		self.debug_log.info('Ending dataset serialization')


if __name__ == "__main__":
	
	if len(sys.argv) < 4:
		print('usage:', sys.argv[0], '<CSV_DATASET>', '<INPUT_DIM>', '<OUTDIR>')
		sys.exit(1)
		
	csv_dataset = sys.argv[1]
	input_dim = sys.argv[2]
	outdir = sys.argv[3]
	
	dataset_preparation = DatasetPreparation(csv_dataset, input_dim, outdir)
	dataset_preparation.prepare_dataset()
	dataset_preparation.write_trace_csv_file()
	dataset_preparation.serialize_dataset()
