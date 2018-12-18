#!/bin/bash

# If tensorflow has been installed in a virtualenv, please uncomment the following line
# source ~/tensorflow/bin/activate

#MODIFICATO
if [[ $# -lt 5 ]];
	then echo "usage $0 <FIRST_M_BYTES_EXTRACTOR> <PICKLE_DIR> <NUM_BYTES> <CSV_FILE> <OUT_DIR>";
	exit 1;
fi

FIRST_M_BYTES_EXTRACTOR=$1
PICKLE_DIR=$2
NUM_BYTES=$3
CSV_FILE=$4
OUT_FOLDER=$5       #directory nuova cartella di output

PYTHON=python


# Check if the directory $PCAPS_DIR does not exists
if [[ ! -d $PICKLE_DIR ]]; then
	echo "$PICKLE_DIR directory does not exist."
	exit 1;
fi



#NUOVO
#creo la cartella contenente i file di output
if [ -d $OUT_FOLDER ]; then
    echo "$OUT_FOLDER exists."
else
    mkdir -p $OUT_FOLDER
fi


#AGGIUNTO
#OUT_FOLDER=$(dirname  $PICKLE_DIR)/TRACE_LEVEL_DATASET

# se non esiste crea cartella contenente i file finali #MODIFICATO
#if [[ ! -e $OUT_FOLDER ]]; then
#	mkdir $OUT_FOLDER
#fi



# Output .csv file is deleted if it exists #MODIFICATO
if [ -f  $OUT_FOLDER/$CSV_FILE ]; then
    echo "$CSV_FILE exists."
    rm -i $OUT_FOLDER/$CSV_FILE
    echo "$OUT_FOLDER/$CSV_FILE successfully deleted."
fi



#MODIFICATO
PICKLE_FILES=$(find $PICKLE_DIR -iname '*wang2017endtoend_L7_biflows_multi-task.pickle')


#MODIFICATO
for PICKLE_FILE in $PICKLE_FILES;
do

    #PCAP_NAME  extraction
    PCAP_NAME=$(basename  $PICKLE_FILE)
    #elimina estensione *wang2017endtoend_L7_biflows_multi-task.pickle
	PCAP_NAME=${PCAP_NAME%_wang2017endtoend_L7_biflows_multi-task.pickle}
	#aggiunge estensione .pcap
	#SEARCH_PCAP=$PCAP_NAME.pcap*
	#echo $SEARCH_PCAP
	#crea directory del file csv finale
    OUT_CSV=$OUT_FOLDER/$CSV_FILE

    # Label for multi-task VPN-nonVPN dataset
    LABEL=$(basename $PCAP_NAME | cut -f1-3 -d'_' ) 
    echo "Processing capture: "$PCAP_NAME" --- Assigned label: "$LABEL

    #MODIFICATO PICKLE_DIR->OUT_FOLDER
	$PYTHON $FIRST_M_BYTES_EXTRACTOR $PICKLE_FILE $OUT_FOLDER $NUM_BYTES $LABEL $OUT_CSV $PCAP_NAME

done



