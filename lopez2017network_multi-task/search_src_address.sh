#!/usr/bin/env bash

# If tensorflow has been installed in a virtualenv, please uncomment the following line
# source ~/tensorflow/bin/activate

if [[ $# -lt 3 ]];
	then echo "usage $0 <SRC_ADDRESS_EXTRACTOR> <PCAPS_DIR> <OUTDIR>";
	exit 1;
fi

SRC_ADDRESS_EXTRACTOR=$1
PCAPS_DIR=$2
OUTDIR=$3   #directory nuova cartella di output
PYTHON=python




# Check if the directory exixts
if [[ !  -d $PCAPS_DIR ]]; then
	echo "$PCAPS_DIR directory does not exist."
	exit 1;
fi


#creo la cartella contenente i file di output
if [ -d $OUTDIR ]; then
    echo "$OUTDIR exists."
else
    mkdir -p $OUTDIR
fi

filename=$(find $OUTDIR -iname 'src_address.txt')

if [ ! -f $filename ]; then
  echo "src_address file does not exist"
else
  rm $filename
fi
   

for PCAP_FILE in $PCAPS_DIR/*.pcap*
do

	if ! [[ ($(basename $PCAP_FILE) == "nonVPN_FileTransfer_ftps_down_1.pcap"  )|| ($(basename $PCAP_FILE) == "nonVPN_FileTransfer_ftps_down_2.pcap"  ) ]] ; then
	
	# src_address extraction
	$PYTHON $SRC_ADDRESS_EXTRACTOR $PCAP_FILE $OUTDIR
	
	fi 
done


filename1=$OUTDIR"whois.txt"

if [ ! -f $filename1 ]; then
  echo "whois file does not exist"
else
  rm $filename1
fi

# Lettura da file degli indirizzi IP

righe=$(wc -l $filename | awk '{print $1}')
echo $righe

riga=0

while [ $riga -lt $righe ]; do

	let riga=riga+1
	current=$(head -$riga $filename | tail -1)
	search_address=$(echo $current| cut -d' ' -f 1)
	#echo $search_address
	echo $search_address >> $filename1
	whois $search_address |grep -i "orgname\|org-name\|descr\|NetName" >> $filename1
	echo "----------------------------" >>$filename1
	echo -e '\r' >> $filename1
	echo -e '\r' >> $filename1

done

