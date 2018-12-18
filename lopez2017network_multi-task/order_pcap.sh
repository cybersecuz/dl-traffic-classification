#!/usr/bin/env bash


if [[ $# -lt 1 ]];
	then echo "usage $0  <PCAP_DIR> <OUT_DIR>";
	exit 1;
fi

PCAP_DIR=$1 # Directory containing .pcap traces
OUT_DIR=$2

# Check if the directory exixts
if [[ ! -d $PCAP_DIR ]]; then
	echo "$PCAP_DIR directory does not exist."
	exit 1;
fi

if [ -d $OUT_DIR ]; then
    echo "$OUT_DIR exists."
else
    mkdir -p $OUT_DIR
fi 

PCAP_FILES=$(find $PCAP_DIR -iname '*.pcap*')

for PCAP_FILE in $PCAP_FILES
    do
        PCAP=$(basename $PCAP_FILE)
        NEW_PCAP=$OUT_DIR/$PCAP
	#echo $NEW_PCAP
        if [ -f $NEW_PCAP ]; then
            echo  "$NEW_PCAP exists."
            rm -rf $NEW_PCAP
            echo  "$NEW_PCAP successfully deleted."
        fi

        reordercap $PCAP_FILE $NEW_PCAP
    done
