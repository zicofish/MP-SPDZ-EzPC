#!/bin/bash

NUMIMAGES=1000
NUMPROCESSES=2
PERPROCESSIMAGES=$((NUMIMAGES / NUMPROCESSES))
RUNSERIAL=false #Change this to "true" to run the first NUMIMAGES.
				#	"false" means it will pick up a random subset.
				#	NOTE: In that case, NUMIMAGES should equal #(lines in the random subset idx file).

RANDOMSUBSETFILEIDX="../../../HelperScripts/ImageNet_ValData/Random_Img_Idx.txt"
RANDOMSUBSETFILELABELS="../../../HelperScripts/ImageNet_ValData/Random_Img_Labels.txt"

# Load paths for imagenet dataset from config file - paths.config
# NOTE: None of the paths mentioned in Paths.config should have a '/' at their end.
. ./Paths.config
declare -a pids

imgFolderName=$(realpath "$imgFolderName")
bboxFolderName=$(realpath "$bboxFolderName")
preProcessedImgSaveDirName=$(realpath "$preProcessedImgSaveDirName")

echo -e "********Running preprocessing for ResNet********"
echo -e "ImgFolderName - $imgFolderName"
echo -e "bboxFolderName - $bboxFolderName"
echo -e "Image name prefix - $imgFilePrefix"
echo -e "PreProcessed images save folder - $preProcessedImgSaveDirName"

cd ../PreProcessingImages
if [ "$RUNSERIAL" = false ];
then
	# Check if the file denoting random subset is already present where its supposed to
	if [ ! -f "$RANDOMSUBSETFILEIDX" ] || [ ! -f "$RANDOMSUBSETFILELABELS" ]; then
	    echo -e "Files to indicate random subset to be chosen not present. Run ../../../HelperScripts/Random_Image_Selection.py."
	    exit 1
	fi
fi

for ((curProcessNum=0;curProcessNum<$NUMPROCESSES;curProcessNum++)); do
	curImgStartNum=$(( curProcessNum*PERPROCESSIMAGES + 1 ))
	endImgNum=$(( curImgStartNum + PERPROCESSIMAGES ))
	if (( curProcessNum == $NUMPROCESSES-1 ));then
		endImgNum=$(( NUMIMAGES + 1 ))
	fi
	if [ "$RUNSERIAL" = false ];
	then
		python3 DenseNet_preprocess_main.py "$imgFolderName" "$bboxFolderName" "$imgFilePrefix" "$preProcessedImgSaveDirName" "$curImgStartNum" "$endImgNum" "$RANDOMSUBSETFILEIDX" &
	else
		python3 DenseNet_preprocess_main.py "$imgFolderName" "$bboxFolderName" "$imgFilePrefix" "$preProcessedImgSaveDirName" "$curImgStartNum" "$endImgNum" &
	fi
	pids+=($!)
done

echo -e "--->>>All processes started. Now going into waiting for each process.<<<---"
for pid in ${pids[*]}; do
    wait $pid
done
echo -e "--->>>All processes completed.<<<---"
