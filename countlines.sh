#!/bin/bash

if [[ $# -lt 1 ]];
    then echo "usage $0 <CSV_DIR>";
    exit 1;
fi

FILES_DIR=$1

#MODIFICATO
FILES=$(find $FILES_DIR -iname '*.csv')


#MODIFICATO
for CSV in $FILES;
do

wc -l $CSV

done
