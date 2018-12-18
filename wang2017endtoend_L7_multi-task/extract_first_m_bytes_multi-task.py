#!/usr/bin/env python

import scapy
import os
import os.path
from scapy.all import *
import pickle
import logging
import sys
from sys import argv
import cPickle as pickle
import csv
from collections import Counter



class CustomExtractFirstMBytesException(Exception):
	def __init__(self, error_message):
		self.error_message = error_message
	def __str__(self):
		return str(self.error_message)



class ExtractFirstMBytes(object):
	
	def __init__(self, pickle_filename, outdir, bytes_number, trace_label, global_csv_filename,pcap_name):
		'''
		ExtractFirstMBytes class constructor
		:param pickle_filename: pickle filename given as input
		:param outdir: output directory of the .csv file associated with the pickle file analyzed
		:param bytes_number: number of bytes extracted from the payload
		:param trace_label: label per traccia da assegnare al biflusso
		:param global_csv_filename: filename of the global .csv file
		:param pcap_name: nome del file .pickle analizzato
		'''

		self.biflow_mbyte = {}
		self.biflow_extract = {}
		self.biflow_label = {}

		self.bytes_number = bytes_number
		self.outdir = outdir
		self.global_csv_filename = global_csv_filename
		self.trace_label = trace_label
		self.pcap_name = pcap_name

		
		if os.path.isfile(pickle_filename):
			self.pickle_filename = pickle_filename
		else:
			error_msg = 'Error: pickle file does not exist'
			raise CustomExtractFirstMBytesException(error_msg)

	
	def get_trace_label(self):
		if self.trace_label:
			return self.trace_label
		else:
			error_msg = 'Error: trace_label not assigned yet'
			raise CustomExtractFirstMBytesException(error_msg)

	
	
	def biflow_extraction(self):
		'''
		Extracts the first M bytes of each biflow and saves them
		'''
		
		for quintuple in self.biflow_extract:
			biflows_b = {}
			
			biflow = list(''.join(self.biflow_extract[quintuple]))
			# print biflow
			
			self.byte_extraction(biflow)
			
			biflows_b[quintuple] = biflow
			# print biflows_b
			
			if biflows_b.items() not in self.biflow_mbyte.items():
				self.biflow_mbyte[quintuple]= biflows_b[quintuple]
		
		# print self.biflow_mbyte
	
	
	def byte_extraction(self, biflow):
		'''
		Extracts the first M bytes of the payload
		'''
		
		biflow_payload_len = len(biflow)
		if biflow_payload_len < self.bytes_number:
			add_0_padding = str("\x00" * (self.bytes_number - biflow_payload_len))
			biflow.extend((add_0_padding))
			"".join(biflow)
			# print biflow
			# print len(biflow)
		
		elif biflow_payload_len > self.bytes_number:
			del biflow[self.bytes_number:]
			# print biflow
			# print len(biflow)
	
	
	def decimal_conv(self):
		'''
		Perform decimal conversion of payload bytes
		'''
		
		for quintuple in self.biflow_mbyte:
			i_payload = self.biflow_mbyte[quintuple]
			# print len(i_payload)
			# print i_payload
			
			self.biflow_mbyte[quintuple] = list(bytearray(i_payload))
		
		# print self.biflow_mbyte
	
	
	def write_trace_level_csv_file(self):
		'''
		Saves the first M bytes of the biflows of each capture in CSV format and in the global_csv_filename (if trace-level labels are used)
		'''


		csv_dir = '%s/%s_multi-class' % (self.outdir, self.pcap_name)
		csv_filename = '%s/%s_trace_level_labels_biflows__wang2017endtoend_L7_biflows_multi-class.csv' % (csv_dir, self.pcap_name)
		
		os.makedirs(csv_dir) if not os.path.exists(csv_dir) else None
		os.remove(csv_filename) if os.path.exists(csv_filename) else None
		self.vpn_labels,self.class_labels,self.app_labels = (self.trace_label.split ("_"))		

		with open(csv_filename, 'wb') as csv_file:
			with open(self.global_csv_filename, 'a') as global_csv_file:
				for quintuple in self.biflow_mbyte:
					writer = csv.writer(csv_file, delimiter=',')
					wout = csv.writer(global_csv_file, delimiter=',')
					
					writer.writerows([self.biflow_mbyte[quintuple] + [self.vpn_labels] + [self.class_labels] + [self.app_labels]])
					wout.writerows([self.biflow_mbyte[quintuple] + [self.vpn_labels] + [self.class_labels] + [self.app_labels]])
	
	
	def deserialize_biflow_pickle(self):
		'''
		Deserializes biflows as python objects from the .pickle file to extract the first M bytes
		'''
		
		with open(self.pickle_filename, 'rb') as deserialize_biflow:
			self.biflow_extract = pickle.load(deserialize_biflow)



if __name__ == "__main__":
	
	if len(sys.argv) < 7: 
		print 'Usage:', sys.argv[0], '<PICKLE_FILENAME>', '<PICKLE_DIRECTORY>', '<BYTES_NUMBER>', '<TRACE_LABEL>', '<GLOBAL_CSV_FILENAME>',  '<PCAP_NAME>'
		sys.exit(1)
	
	pickle_filename = sys.argv[1]
	outdir = sys.argv[2]
	bytes_number = int(sys.argv[3])
	trace_label = sys.argv[4]
	global_csv_filename = sys.argv[5]
	pcap_name = sys.argv[6]


	print 'Starting extraction'
	print 'Analyzing: ' + pickle_filename
	extract_first_m_bytes = ExtractFirstMBytes(pickle_filename, outdir, bytes_number, trace_label, global_csv_filename,pcap_name) 
	
	
	# Extracts trace-level label	
	if extract_first_m_bytes.get_trace_label() != 'NA':
		extract_first_m_bytes.deserialize_biflow_pickle()
		extract_first_m_bytes.biflow_extraction()
		extract_first_m_bytes.decimal_conv()
		extract_first_m_bytes.write_trace_level_csv_file()



