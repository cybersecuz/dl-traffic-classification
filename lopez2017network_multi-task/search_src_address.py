import os, os.path, sys
import scapy
from scapy.all import *
import json
import pickle
import cPickle as pickle
from collections import Counter
import numpy
#import dill as pickle

import numpy as np

class CustomMatrixExtractionException(Exception):
	def __init__(self, error_message):
		self.error_message = error_message
	def __str__(self):
		return str(self.error_message)


class Biflow(object):


    def __init__(self, extracted_pcap, out_pickle):
        '''
        MatrixExtraction class constructor
        :param self:
        :param extracted_pcap: traffic trace in .pcap format
        :param pcap_label: label (timestamp) of the pcap file
        :param out_pickle: pickle file updated with the matrix extracted from the current extracted_pcap
        '''

        self.biflow = {}
        self.all_biflows = {}
        self.src_address = ''
        self.address_list = []

        self.m_packets = 32 #modificato 20
        self.n_features = 6
        self.lopez_biflow = {}
        self.out_pickle = out_pickle

        if os.path.isfile(extracted_pcap):
            self.extracted_pcap = extracted_pcap
        else:
            raise CustomMatrixExtractionException('Error: pcap file does not exist')

    def biflows_extraction(self):
        '''
        Extract and save in a dict the TCP payloads of each biflow
        '''

        pcap_trace = rdpcap(self.extracted_pcap)

        # Payloads extraction
        for packet in pcap_trace:
            try:
                if packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP)):
                    if not (packet.haslayer(NTP) or packet.haslayer(DNS) or packet.sport == 5353 or packet.dport == 5353):

                        # packet.show() # useful to show the packet

                        # Both TCP and UDP packets with and without payload are considered
                        layer_src_ip = packet[IP].src
                        self.address_list.append(layer_src_ip)

            except:
                traceback.print_exc(file=sys.stdout)
                logging.error('Biflow extraction failed', exc_info=True)
                continue




    def extract_most_common_address(self):
            '''
            Extracts the most-common SRC_IP address 
            '''

            count_source_address = Counter([source_address for source_address in self.address_list])
	    # Extract the most-common of the source addresses
            most_common = count_source_address.most_common(4)
	    print most_common
	    dat_filename = '%s/src_address.txt' % (self.out_pickle)
	    with open(dat_filename, 'a') as write_dat:

		    numpy.savetxt(write_dat, most_common, fmt='%s')
			
					
			
		    







if __name__ == "__main__":

    if len(sys.argv) < 3:
        print 'Usage: ' + sys.argv[0] + ' <PCAP_FILE> '  + ' <OUTDIR>'
        sys.exit(1)

    extracted_pcap = sys.argv[1]
    out_pickle = sys.argv[2]


    print 'Starting extraction'
    print 'Analyzing: ' + extracted_pcap
    src_address_extraction = Biflow(extracted_pcap, out_pickle)


    src_address_extraction.biflows_extraction()
    src_address_extraction.extract_most_common_address()

