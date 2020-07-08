DATASET_NAME=$1
RAW_DATADIR=$2

function download_25m {
	echo "Download ml-25m"
	cd ${RAW_DATADIR}
	curl -O http://files.grouplens.org/datasets/movielens/ml-25m.zip
	cd -
}

function download_1m {
	echo "Downloading ml-1m"
	cd ${RAW_DATADIR}
	curl -O http://files.grouplens.org/datasets/movielens/ml-1m.zip
        cd -
}


if [[ ${DATASET_NAME} == "ml-1m" ]]
then
	download_1m
elif [[ ${DATASET_NAME} == "ml-25m" ]]
then
    download_25m
else
	echo "Unsupported dataset name: $DATASET_NAME"
	exit 1
fi
