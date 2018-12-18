#!/usr/bin/env bash

# If tensorflow has been installed in a virtualenv, please uncomment the following line
source ~/tensorflow/bin/activate

if [[ $# -lt 4 ]];#MODIFICATO
	then echo "usage $0 <MATRIX_EXTRACTOR>  <FILES_DIR> <OUT_PICKLE_NAME> <OUTDIR>";
	exit 1;
fi

MATRIX_EXTRACTOR=$1
FILES_DIR=$2			# Directory containing tutti i file di uscita /HUAWEI_OUT
OUT_PICKLE_NAME=$3      # nome da assegnare al file.pickle finale
OUT_FOLDER=$4           #directory nuova cartella di output

PYTHON=python



# Check if the directory exixts
if [[ ! -d $FILES_DIR ]]; then
	echo "$FILES_DIR directory does not exist."
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
#OUT_FOLDER=$(dirname  $FILES_DIR)/TRACE_LEVEL_DATASET

# se non esiste crea cartella contenente i file finali
#if [[ ! -e $OUT_FOLDER ]]; then
#        mkdir $OUT_FOLDER
#fi



#percorso + nome del pickle di output (contentente tutte le matrici di lopez) MODIFICATO
if [ -f $OUT_FOLDER/$OUT_PICKLE_NAME ]; then
	echo  "$OUT_FOLDER/$OUT_PICKLE_NAME  exists."
	rm -i $OUT_FOLDER/$OUT_PICKLE_NAME
	echo  "$OUT_FOLDER/$OUT_PICKLE_NAME  successfully deleted."
fi



#MODIFICATO
PICKLE_FILES=$(find $FILES_DIR -iname '*lopez2017applications_biflows_multi-class.pickle')



for PICKLE_FILE in $PICKLE_FILES
do
        #MODIFICATO
        #PCAP_NAME  extraction
        PCAP_NAME=$(basename  $PICKLE_FILE)
        #elimina estensione lopez_biflows_multi-class.pickle
	    PCAP_NAME=${PCAP_NAME%_lopez2017applications_biflows_multi-class.pickle}
	    #SEARCH_PCAP=$PCAP_NAME.pcap*
	    #echo $SEARCH_PCAP
        OUT_PICKLE=$OUT_FOLDER/$OUT_PICKLE_NAME
        echo $OUT_PICKLE

	    # Label for multi-class Huawei dataset
        LABEL=$(basename $PCAP_NAME | cut -f1-3 -d'_' )
        echo "Processing capture: "$PCAP_NAME" --- Assigned label: "$LABEL




        $PYTHON $MATRIX_EXTRACTOR $PICKLE_FILE $LABEL $OUT_PICKLE

done
