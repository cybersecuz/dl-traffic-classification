#!/usr/bin/env bash

# source ~/tensorflow/bin/activate

if [[ $# -lt 5 ]];
	then echo "usage $0 <EXTRACTOR> <CSV_DIR> <SEARCH_CSV> <INPUT_DIM> <OUT_DIR>";
	exit 1;
fi

EXTRACTOR=$1
CSV=$2
SEARCH_CSV=$3 #file .csv da cercare
INPUT_DIM=$4
OUT_DIR=$5

PYTHON=python3

#controllo se la directory non esiste
 if [[ !  -d $CSV ]]; then
    echo " directory does not exist "
    exit 1;
 fi



CSV_DATASET=$(find $CSV  -iname $SEARCH_CSV)

if [[ ! -f $CSV_DATASET ]]; then
    echo  "csv file does not exist"
fi


# Estrazione directory
# OUT_DIR=$(dirname $CSV_DATASET)


$PYTHON  $EXTRACTOR $CSV_DATASET  $INPUT_DIM $OUT_DIR
