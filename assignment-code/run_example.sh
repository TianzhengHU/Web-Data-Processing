#!/bin/sh
echo "Processing webpages ..."
python starter_code.py data/warcs/sample.warc.gz > sample_predictions.tsv
echo "Computing the scores ..."
python score.py predictions.txt sample_predictions.tsv ENTITY
