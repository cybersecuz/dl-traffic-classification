#!/usr/bin/env python

import logging
import os.path
import pickle
import sys
from scapy.all import *
from scapy.layers.inet import IP
from scapy.layers.inet import TCP



class CustomBiflowException(Exception):
	def __init__(self, error_message):
		self.error_message = error_message
	def __str__(self):
		return str(self.error_message)



class Biflow(object):
	
	def __init__(self, extract_pcap, pcap_label, directory_out):
		'''
		Biflow constructor
		:param self:
		:param extract_pcap: traffic trace in .pcap format
		:param pcap_label: label (timestamp) of the pcap file
		:param directory_out: output directory in which the file .pickle will be saved
		'''
		
		self.biflow = {}
		self.directory_out = directory_out
		self.pcap_label = pcap_label
		
		if os.path.isfile(extract_pcap):
			self.extract_pcap = extract_pcap
		else:
			raise CustomBiflowException('Error: pcap file does not exist')
	
	
	def biflows_extraction(self):
		'''
		Extract and save in a dict the TCP and UDP payloads of each biflow
		'''
		
		pcap_trace = rdpcap(self.extract_pcap)
		
		# Payloads extraction
		for packet in pcap_trace:
			try:
				# XXX: both TCP and UDP packets are considered
				if packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP)) and packet.haslayer(Raw):
					# NTP, DNS, and mDNS packets are not considered
					if not (packet.haslayer(NTP) or packet.haslayer(DNS) or packet.sport == 5353 or packet.dport == 5353 or packet[IP].src == '224.0.0.252' or packet[IP].dst == '224.0.0.252'):
						# packet.show() # useful to show the packet
						
						'''
						# To print plaintext data (HTTP header)
						if packet.haslayer(Raw) and TCP in packet and packet[TCP].dport == 80 and packet[TCP].load.startswith("GET"):
							print packet[TCP].load ( print payload data ) 
						'''
						
						# Only with TCP and UDP packets with non-zero payload are considered
						layer_src_ip = packet[IP].src
						layer_dst_ip = packet[IP].dst
						# print layer_dest_ip
						
						layer_src_port = packet.sport
						layer_dst_port = packet.dport
						# print layer_dst_port
						
						payload_data = packet.getlayer(Raw).load
						# print len(payload_data)
						# print payload_data
						
						# Information extraction
						src_socket = '%s,%s' % (layer_src_ip, layer_src_port)
						# print src_socket
						dst_socket = '%s,%s' % (layer_dst_ip, layer_dst_port)
						# print dest_socket
						proto = packet[IP].proto
						# print proto
						
						# 5-tuples in both directions are created
						quintuple = '%s,%s,%s' % (src_socket, dst_socket, proto)
						# print quintuple
						inverse_quintuple = '%s,%s,%s' % (dst_socket, src_socket, proto)
						# print inverse_quintuple
						
						# raise CustomBiflowException('Layer does not exist')
						if quintuple not in self.biflow and inverse_quintuple not in self.biflow:
							self.biflow[quintuple] = []
							# self.biflow[quintuple].append('{}'.format(hexdump(data_payload)))
							# self.biflow[quintuple].append('%s' % (str(data_payload)))
							self.biflow[quintuple].append('{}'.format(str(payload_data)))
							# print self.biflow
						elif quintuple in self.biflow:
							self.biflow[quintuple].append('{}'.format(str(payload_data)))
							# print self.biflow
						elif inverse_quintuple in self.biflow:
							self.biflow[inverse_quintuple].append('{}'.format(str(payload_data)))
							# print self.biflow
						else:
							logging.error('Packet does not belong to any biflow : %s --> %s' % (src_socket, dst_socket), exc_info=True)
							# print self.biflow
			except:
				traceback.print_exc(file=sys.stdout)
				# logging.error('Packet does not belong to any biflow : %s --> %s' % (src_socket, dst_socket), exc_info=True)
				continue
		
		# print len(self.biflows)
	
	
	def serialize_biflow_pickle(self):
		'''
		Serialize extracted biflows as python objects to later perform classification using DL networks
		'''

		pickle_directory = '%s/%s_multi-task' % (self.directory_out, self.pcap_label)
		pickle_filename = '%s/%s_wang2017endtoend_L7_biflows_multi-task.pickle' % (pickle_directory, self.pcap_label)
		
		os.makedirs(pickle_directory) if not os.path.exists(pickle_directory) else None
		os.remove(pickle_filename) if os.path.exists(pickle_filename) else None
		
		with open(pickle_filename, 'wb') as serialize_biflow:
			pickle.dump(self.biflow, serialize_biflow, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
	
	if len(sys.argv) < 4:
		print 'Usage:', sys.argv[0], '<PCAP_FILE>', '<PCAP_LABEL>', '<OUTDIR>'
		sys.exit(1)
	
	extract_pcap = sys.argv[1]
	pcap_label = sys.argv[2]
	directory_out = sys.argv[3]
	
	print 'Starting extraction'
	print 'Analyzing: ' + extract_pcap
	
	extr = Biflow(extract_pcap, pcap_label, directory_out)
	
	extr.biflows_extraction()
	extr.serialize_biflow_pickle()



