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


    def __init__(self, extracted_pcap, pcap_label,out_pickle):
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
        self.pcap_label= pcap_label

        if os.path.isfile(extracted_pcap):
            self.extracted_pcap = extracted_pcap
        else:
            raise CustomMatrixExtractionException('Error: pcap file does not exist')

    def biflows_extraction(self):
        '''
        Extract and save in a dict the TCP payloads of each biflow
        '''
		# pcap_trace = rdpcap(self.extract_pcap)
        pcap_reader = PcapReader(self.extracted_pcap)

		# Payloads extraction
        for packet in pcap_reader:
            try:
                if packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP)):
                    if not ( packet.haslayer(NTP) or packet.haslayer(DNS) or packet.sport == 5353 or packet.dport == 5353 or packet[IP].src == '224.0.0.252' or packet[IP].dst == '224.0.0.252'):

                        # packet.show() # useful to show the packet

                        # Both TCP and UDP packets with and without payload are considered
                        layer_src_ip = packet[IP].src
                        layer_dst_ip = packet[IP].dst
                        # print layer_dest_ip

                        layer_src_tcp = packet.sport
                        layer_dst_tcp = packet.dport
                        # print layer_dst_tcp

                        self.address_list.append(layer_src_ip)

                        # Information extraction
                        src_socket = '%s,%s' % (layer_src_ip, layer_src_tcp)
                        # print src_socket
                        dst_socket = '%s,%s' % (layer_dst_ip, layer_dst_tcp)
                        # print dest_socket
                        proto = packet[IP].proto
                        # print proto


                        # 5-tuples in both directions are created
                        quintuple = '%s,%s,%s' % (src_socket, dst_socket, proto)
                        # print quintuple
                        inverse_quintuple = '%s,%s,%s' % (dst_socket, src_socket, proto)
                        # print inverse_quintuple

                        if quintuple not in self.all_biflows and inverse_quintuple not in self.all_biflows:
                            self.all_biflows[quintuple] = []
                            self.all_biflows[quintuple].append(packet)
                        elif quintuple in self.all_biflows:
                            if len(self.all_biflows[quintuple])< self.m_packets:
                                self.all_biflows[quintuple].append(packet)
                        elif inverse_quintuple in self.all_biflows:
                            if len(self.all_biflows[inverse_quintuple]) < self.m_packets:
                                self.all_biflows[inverse_quintuple].append(packet)
                        else:
                            logging.error('Packet does not belong to any biflow : %s --> %s' % (src_socket, dst_socket),
                                          exc_info=True)

            except:
                traceback.print_exc(file=sys.stdout)
                logging.error('Biflow extraction failed', exc_info=True)
                continue




    def extract_most_common_address(self):
            '''
            Extracts the most-common SRC_IP address (it should be a 192.168.*.* address)
            '''

            count_source_address = Counter([source_address for source_address in self.address_list \
            if (source_address.startswith('192.168') or \
	    (source_address.startswith('172.') and 16 <= int(source_address.split('.')[1]) <= 32) or \
	    source_address.startswith('10.'))])

	    # Extract the most-common of the source addresses
            most_common = count_source_address.most_common(1)
            #print most_common
            if len(most_common) == 0:
		    count_source_address = Counter([source_address for source_address in self.address_list \
		    if (source_address.startswith('84.235.54') or \
		    source_address.startswith('131.204.240'))])
		    most_common = count_source_address.most_common(1)


            for src_ip, num_of_occurrence in most_common:
                self.src_address = str(src_ip)
                print 'Most-common private IP address: ' + self.src_address



    def packets_info_extraction(self):
            '''
            Extracts the N=6 features from the first M=20 packets of each biflow as in lopez2017network
            '''

            print len(self.all_biflows.keys())
            for quintuple in self.all_biflows:
                self.lopez_biflow[quintuple] = []
                cons_pkt = []

                # Extract the N=6 features from the first M=20 packets
                for pkt in self.all_biflows[quintuple]:

                    # Extract the TCP window size
                    if pkt.haslayer(TCP):
                        window_size = pkt[TCP].window
                    else:
                        window_size = 0

                    # Extract source and destination ports
                    src_port = pkt.sport
                    dst_port = pkt.dport

                    # Extract payload length, if is exists
                    if pkt.haslayer(Raw):
                        payload_length = len(pkt.getlayer(Raw).load)
			# pkt.show()
                    	# print payload_length
                    # payload_length = len(pkt[TCP].payload)
                    else:
                        payload_length = 0
                    	# pkt.show()
                    	# print payload_length

                    # Determine the direction of the packets:
                    #   - 0 if the packet goes from the source to the destination (uplink)
                    #   - 1 if the packet goes from the destination to the source (downlink)
                    if pkt[IP].src == self.src_address:
                        pkt_direction = 0
                    else:
                        pkt_direction = 1

                    # Compute inter-arrival times
                    # Create a list containing the timestamps related to two successive packets
                    cons_pkt.append(pkt.time)
                    if len(cons_pkt) < 2:
                        interarrival_time = 0
                    else:

                        interarrival_time = numpy.diff(cons_pkt)[0]
                        cons_pkt = cons_pkt[1:]
                        #print interarrival_time


                    six_features = [src_port, dst_port, pkt_direction, payload_length, interarrival_time, window_size]
                    self.lopez_biflow[quintuple].append(six_features)


                # Add zero-padding if the number of packets is less than N=20
                # print len(self.biflow[quintuple])
                if len(self.all_biflows[quintuple]) < self.m_packets:
                    add_pkts = self.m_packets - len(self.all_biflows[quintuple])
                    for pkt in range(0, add_pkts):
                        add_padding = ([0] * self.n_features)
                        self.lopez_biflow[quintuple].append(add_padding)
            #print self.lopez_biflow.keys()



    def serialize_biflow_pickle(self):
        '''
        Serialize extracted biflows as python objects to later perform classification using DL networks
        '''

        #pickle_directory = '%s/%s/%s_multi-class/%s' % (self.out_pickle,"HUAWEI_OUT", self.pcap_label,"PICKLE")#MODIFICATO
        pickle_directory = '%s/%s_multi-class' % (self.out_pickle, self.pcap_label)  #NUOVO
        pickle_filename = '%s/%s_lopez2017applications_biflows_multi-class.pickle' % (pickle_directory, self.pcap_label)#MODIFICATO

        os.makedirs(pickle_directory) if not os.path.exists(pickle_directory) else None
        os.remove(pickle_filename) if os.path.exists(pickle_filename) else None



        with open(pickle_filename, 'wb') as serialize_biflow:
            pickle.dump(self.lopez_biflow, serialize_biflow, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":

    if len(sys.argv) < 4:
        print 'Usage: ' + sys.argv[0] + ' <PCAP_FILE> ' + ' <PCAP_LABEL> ' + ' <OUTDIR>'
        sys.exit(1)

    extracted_pcap = sys.argv[1]
    pcap_label = sys.argv[2]
    out_pickle = sys.argv[3]


    print 'Starting extraction'
    print 'Analyzing: ' + extracted_pcap
    biflow_extraction = Biflow(extracted_pcap, pcap_label, out_pickle)


    biflow_extraction.biflows_extraction()
    biflow_extraction.extract_most_common_address()
    biflow_extraction.packets_info_extraction()
    biflow_extraction.serialize_biflow_pickle()
