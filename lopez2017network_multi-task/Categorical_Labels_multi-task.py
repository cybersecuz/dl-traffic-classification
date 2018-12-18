#!/usr/bin/env python
import os
import os.path
#import scapy
#from scapy.all import *
import logging
import sys
from sys import argv
import pickle
#import cPickle as pickle
from collections import Counter
import numpy
import keras, sklearn, numpy
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold




class Matrix(object):

    def __init__(self, pickle_filename, out_directory,dataset_basename):

        '''
            create constructor
            :param self:
            :param pickle_filename: file pickle da analizzare
            :param out_directory: directory in cui salvare il file .pickle finale
            :param dataset_basename: label da assegnare al file csv di uscita
        '''

        self.out_directory = out_directory
        self.dataset_basename = dataset_basename

        self.samples = []
        self.categorical_vpn_labels_extract=[]
        self.categorical_class_labels_extract=[]
        self.categorical_app_labels_extract=[]
        self.allpickle = []
        self.categorical_vpn_labels_allpickle=[]
        self.categorical_class_labels_allpickle=[]
        self.categorical_app_labels_allpickle=[]
        self.out_dir=  os.path.splitext(os.path.dirname(self.out_directory))[0]

        if os.path.isfile(pickle_filename):

            self.pickle_filename = pickle_filename

        else:

            raise CustomBiflowException('error, pcap file does not exist')



    def deserialize_matrix_pickle(self):
        '''
        deserializzo il file pickle per estrarre le informazioni
        '''

        with open(self.pickle_filename, 'rb') as deserialize_matrix:


            self.samples= (pickle.load(deserialize_matrix, encoding='latin-1'))
            #print self.samples
            #self.categorical_labels_extract=(pickle.load(deserialize_matrix, encoding='latin-1'))
            self.categorical_vpn_labels_extract = (pickle.load(deserialize_matrix, encoding='latin-1'))
            self.categorical_class_labels_extract = (pickle.load(deserialize_matrix, encoding='latin-1'))
            self.categorical_app_labels_extract = (pickle.load(deserialize_matrix, encoding='latin-1'))
            # print (self.categorical_vpn_labels_extract)
            # print (self.categorical_class_labels_extract)
            # print (self.categorical_app_labels_extract)



    def prepare_dataset(self):


        self.allpickle=self.samples
        #print numpy.shape(self.allpickle)

        # Convert the labels from a list of strings to categorical values
        label_vpn_encoder = preprocessing.LabelEncoder()
        label_class_encoder = preprocessing.LabelEncoder()
        label_app_encoder = preprocessing.LabelEncoder()
        label_vpn_encoder.fit(self.categorical_vpn_labels_extract)
        label_class_encoder.fit(self.categorical_class_labels_extract)
        label_app_encoder.fit(self.categorical_app_labels_extract)
        self.categorical_vpn_labels_allpickle = label_vpn_encoder.transform(self.categorical_vpn_labels_extract)
        self.categorical_class_labels_allpickle = label_class_encoder.transform(self.categorical_class_labels_extract)
        self.categorical_app_labels_allpickle = label_app_encoder.transform(self.categorical_app_labels_extract)
        # print (self.categorical_vpn_labels_allpickle)
        print (numpy.shape (self.categorical_vpn_labels_allpickle))
        # print (self.categorical_class_labels_allpickle)
        # print (numpy.shape (self.categorical_class_labels_allpickle))
        # print (self.categorical_app_labels_allpickle)
        # print (numpy.shape (self.categorical_app_labels_allpickle))



    def write_trace_csv_file(self):
        '''
        Saves the first M bytes of the biflows of each capture in CSV format and in the global_csv_filename
        '''
        import csv

        csv_filename = '%s/%s_categorical_biflows.csv' % (self.out_dir, self.dataset_basename)

        os.remove(csv_filename) if os.path.exists(csv_filename) else None

        with open(csv_filename, 'wb') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')

            writer.writerows(zip(self.categorical_labels_allpickle, self.categorical_labels_extract))



    def serialize_all_pickle(self):

        # serializzo gli oggetti python in un flusso di caratteri

        pickle_filename = self.out_directory
        with open(pickle_filename, 'wb') as serialize_dataset:


                pickle.dump(self.allpickle, serialize_dataset, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.categorical_vpn_labels_allpickle, serialize_dataset, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.categorical_class_labels_allpickle, serialize_dataset, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.categorical_app_labels_allpickle, serialize_dataset, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":

    if len(sys.argv) < 4:
        print ('Usage:' + sys.argv[0] + '<PickleFile>' + '<Out_Directory>'+'<dataset_basename>')
        sys.exit(1)


pickle_filename = sys.argv[1]
out_directory = sys.argv[2]
dataset_basename= sys.argv[3]


print ("start extraction")
print ('analyzing: ' + pickle_filename)
extr = Matrix(pickle_filename,out_directory,dataset_basename)

extr.deserialize_matrix_pickle()
extr.prepare_dataset()
#extr.write_trace_csv_file()
extr.serialize_all_pickle()
