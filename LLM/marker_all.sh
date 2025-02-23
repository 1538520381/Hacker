#!/bin/bash

# This script will run the marker.sh script on all the pdf

# sudo su
# conda activate LLM
# echo "Conda environment activated"

# Get the directory of the script
# dirname : strip last component from file name
# realpath : print the resolved file name
# $0 : the path to the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# # 行业
PDFS_LOAD_PATH="$SCRIPT_DIR/RAG"
PDFS_OUTPUT_BASEPATH="$SCRIPT_DIR/result_hangye"

mkdir -p $PDFS_OUTPUT_BASEPATH

marker $PDFS_LOAD_PATH --output_dir $PDFS_OUTPUT_BASEPATH --output_format markdown --disable_image_extraction --workers 1

# # 公司
#IFS=$'\n' 
# for COMPANY in $(ls $PDFS_LOAD_PATH); do
#     echo "Processing $COMPANY"
#     PDFS_OUTPUT_PATH="$PDFS_OUTPUT_BASEPATH/$COMPANY"
#     mkdir -p $PDFS_OUTPUT_PATH
#     marker $PDFS_LOAD_PATH/$COMPANY --output_dir $PDFS_OUTPUT_PATH --output_format markdown --disable_image_extraction --workers 2
#     #$SCRIPT_DIR/marker.sh $PDFS_PATH/$pdf
# done

echo "All PDFs have been processed"