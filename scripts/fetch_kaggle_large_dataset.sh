#!/usr/bin/env bash

TRAINING_DATA_URL="vbookshelf/v2-plant-seedlings-dataset"
NOW=$(date)

mkdir -p model_package/cnn_model/datasets/
kaggle datasets download -d $TRAINING_DATA_URL -p model_package/cnn_model/datasets/ && \
unzip model_package/cnn_model/datasets/v2-plant-seedlings-dataset.zip -d model_package/cnn_model/datasets/v2-plant-seedlings-dataset && \
echo $TRAINING_DATA_URL 'retrieved on:' $NOW > model_package/cnn_model/datasets/training_data_reference.txt && \
mv -v model_package/cnn_model/datasets/v2-plant-seedlings-dataset/Shepherd*Purse model_package/cnn_model/datasets/v2-plant-seedlings-dataset/"Shepherds Purse"
rm -rf model_package/cnn_model/datasets/v2-plant-seedlings-dataset/nonsegmentedv2/