#!/usr/bin/env bash

# If tensorflow has been installed in a virtualenv, please uncomment the following line
# source ~/tensorflow/bin/activate

if [[ $# -lt 3 ]];
	then echo "usage $0 <BIFLOW_EXTRACTOR> <PCAPS_DIR> <OUTDIR>";
	exit 1;
fi

BIFLOW_EXTRACTOR=$1
PCAPS_DIR=$2
OUTDIR=$3   #directory nuova cartella di output

PYTHON=python


# Check if the directory exixts
if [[ !  -d $PCAPS_DIR ]]; then
	echo "$PCAPS_DIR directory does not exist."
	exit 1;
fi


#NUOVO
#creo la cartella contenente i file di output
if [ -d $OUTDIR ]; then
    echo "$OUTDIR exists."
else
    mkdir -p $OUTDIR
fi


#MODIFICATO
for PCAP_FILE in $PCAPS_DIR/*.pcap*
do
	# Pcap label extraction
	PCAP_LABEL=$(basename $PCAP_FILE)
	#elimina estensione .pcap
	PCAP_LABEL=${PCAP_LABEL%.pcap*}
	echo $PCAP_LABEL

    #OUTDIR=$(dirname $(dirname $PCAP_FILE))
	# Biflow extraction
	$PYTHON $BIFLOW_EXTRACTOR $PCAP_FILE $PCAP_LABEL $OUTDIR
done


