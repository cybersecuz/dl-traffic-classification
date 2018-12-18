#!/usr/bin/env python

import os, os.path, sys
import scapy
from scapy.all import *
import pickle
import logging
import numpy
import cPickle as pickle
from collections import Counter



class CustomMatrixExtractionException(Exception):
	def __init__(self, error_message):
		self.error_message = error_message
	def __str__(self):
		return str(self.error_message)



class MatrixExtraction(object):

		
	def __init__(self, pickle_filename, trace_label, out_pickle):
		'''
		MatrixExtraction class constructor
		:param self:
		:param pickle_filename:pickle filename given as input
		:param trace_label:
		:param out_pickle: pickle file updated with the matrix extracted from the current extracted_pcap
		'''

		self.trace_label = trace_label
		self.out_pickle = out_pickle


		self.lopez_biflow = {}
		self.samples = []
		self.vpn_labels = []
		self.class_labels = []
		self.app_labels = []
		self.all_matrix = []
		self.categorical_vpn_labels_all_matrix = []

		if os.path.isfile(pickle_filename):
			self.pickle_filename = pickle_filename
		else:
			raise CustomMatrixExtractionException('Error: pickle file does not exist')
	
	
	def get_biflow_label(self):
		if self.trace_label:
			return self.trace_label
		else:
			error_msg = 'Error: biflow_label not assigned yet'
			raise CustomMatrixExtractionException(error_msg)
    


	#AGGIUNTO
	def trace_level_label(self):
		print self.trace_label
		for i in range (0,len(self.lopez_biflow.keys())):
			#self.labels.append(self.trace_label)
			self.vpn_labels.append(self.trace_label.split ("_")[0])
			self.class_labels.append(self.trace_label.split ("_")[1])
			self.app_labels.append(self.trace_label.split ("_")[2])
		print len(self.vpn_labels)
		print len(self.class_labels)
		print len(self.app_labels)
		#print (self.vpn_labels)
		#print (self.class_labels)
		#print (self.app_labels)
	
	
	def prepare_dataset(self):
		'''
		Creates a 3D array with shape N x M x biflow_num
		'''
		
		# Create a 3D array adding the information from each biflow
		for quintuple in self.lopez_biflow:
			self.samples.append(self.lopez_biflow[quintuple])
		
		# Merge 3D arrays extracted from each trace
		if self.all_matrix != []:
			self.all_matrix.extend(self.samples)
		else:
			self.all_matrix = self.samples
		
		print numpy.shape(self.samples)
		print numpy.shape(self.all_matrix)
		
		# Merge rhe labels extracted from each trace
		if self.categorical_vpn_labels_all_matrix != []:
			new_vpn_labels = list(self.categorical_vpn_labels_all_matrix) + list(self.vpn_labels)
			new_class_labels = list(self.categorical_class_labels_all_matrix) + list(self.class_labels) 
			new_app_labels = list(self.categorical_app_labels_all_matrix) + list(self.app_labels)
			self.categorical_vpn_labels_all_matrix = new_vpn_labels
			self.categorical_class_labels_all_matrix = new_class_labels
			self.categorical_app_labels_all_matrix = new_app_labels
		else:
			#self.categorical_labels_all_matrix = list(self.labels)
			self.categorical_vpn_labels_all_matrix = list(self.vpn_labels)
			self.categorical_class_labels_all_matrix = list(self.class_labels)
			self.categorical_app_labels_all_matrix = list(self.app_labels)

		#print self.categorical_app_labels_all_matrix
		#print numpy.shape(self.labels)
		print numpy.shape(self.categorical_vpn_labels_all_matrix)
		print numpy.shape(self.categorical_class_labels_all_matrix)
		print numpy.shape(self.categorical_app_labels_all_matrix)



	def deserialize_biflow_pickle(self):
		'''
		Deserializes biflows as python objects from the .pickle file to extract biflows and address_list
		'''

		with open(self.pickle_filename, 'rb') as deserialize_biflow:
			self.lopez_biflow = pickle.load(deserialize_biflow)



	def deserialize_all_matrix(self):
		'''
		Deserializes biflows as python objects from the .pickle file to extract the NxM features
		'''
		
		if os.path.exists(self.out_pickle):
			with open(self.out_pickle, 'rb') as deserialize_matrix:
				self.all_matrix = (pickle.load(deserialize_matrix))
				self.categorical_vpn_labels_all_matrix = (pickle.load(deserialize_matrix))
				self.categorical_class_labels_all_matrix = (pickle.load(deserialize_matrix))
				self.categorical_app_labels_all_matrix = (pickle.load(deserialize_matrix))


	
	
	def serialize_all_matrix(self):
		'''
		Serializes MxN features extracted from each biflow in the .pickle file
		'''
		
		with open(self.out_pickle, 'wb') as serialize_dataset:
		    pickle.dump(self.all_matrix, serialize_dataset, pickle.HIGHEST_PROTOCOL)
		    pickle.dump(self.categorical_vpn_labels_all_matrix, serialize_dataset, pickle.HIGHEST_PROTOCOL)
		    pickle.dump(self.categorical_class_labels_all_matrix, serialize_dataset, pickle.HIGHEST_PROTOCOL)
		    pickle.dump(self.categorical_app_labels_all_matrix, serialize_dataset, pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
	
	if len(sys.argv) < 4:#MODIFICATO
		print 'Usage:', sys.argv[0], '<PICKLE_FILE>', '<TRACE_LABEL>', '<OUT_PICKLE>'
		sys.exit(1)

	#MODIFICATO
	pickle_filename = sys.argv[1]
	trace_label = sys.argv[2]
	out_pickle = sys.argv[3]
	print out_pickle

	
	print 'Starting extraction'
	print 'Analyzing: ' + pickle_filename

	matrix_extraction = MatrixExtraction(pickle_filename, trace_label, out_pickle)#MODIFICATO
	matrix_extraction.deserialize_biflow_pickle()


	# Extracts trace-level label
		
	if matrix_extraction.get_biflow_label() != 'NA':

		matrix_extraction.trace_level_label()
		matrix_extraction.deserialize_all_matrix()
		matrix_extraction.prepare_dataset()
		matrix_extraction.serialize_all_matrix()
	



