#!/bin/bash

BASE_URL="s3://stocktwits-nyu/dataset/v1/data/csv"

DATA_DIR="/scratch/$USER/stocktwits_dataset/csv"

mkdir -p $DATA_DIR

echo "Downloading messages..."
aws s3 sync --no-sign-request \
${BASE_URL}/messages \
${DATA_DIR}/messages

echo "Downloading feature_wo_messages..."
aws s3 sync --no-sign-request \
${BASE_URL}/feature_wo_messages \
${DATA_DIR}/feature_wo_messages

echo "Download complete."
