#!/usr/bin/env bash

source ~/tensorflow/bin/activate




if [[ $# -lt 4 ]];
	then echo "usage $0 <Categorical_Labels> <INPICKLE_DIR> <PICKLENAME_SEARCH> <OUT_PICKLENAME> ";
	exit 1;
fi

Categorical_Labels=$1
DIRECTORY=$2 #directory in cui cercare il file .pickle (/TRACE LEVEL DATASET)
FILENANAME_SEARCH=$3 #nome del file .pickle da cercare
PICKLE_NAME=$DIRECTORY/$4 #nome da assegnare al .pickle in uscita
PYTHON=python3 



#controllo se la directory non esiste
 if [[ !  -d $DIRECTORY ]]; then
        echo " directory does not exist "
        exit 1;

 fi


#percorso + nome del pickle di output categorico da creare MODIFICATO
if [ -f $PICKLE_NAME ]; then
	echo  "$PICKLE_NAME  exists."
	rm -i $PICKLE_NAME
	echo  "$PICKLE_NAME  successfully deleted."
fi

#verifico se il pickle da cercare esiste
directory=$(find $DIRECTORY  -iname $FILENANAME_SEARCH)
if [[ ! -f $directory ]]; then
    echo  "pickle file does not exist"
fi




for PickleFile in $directory

    do

        OUT_LABEL=$(basename $FILENANAME_SEARCH)
        OUT_LABEL=${OUT_LABEL%.pickle}
        echo $OUT_LABEL

	$PYTHON $Categorical_Labels $PickleFile $PICKLE_NAME $OUT_LABEL


    done







