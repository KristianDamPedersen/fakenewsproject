#!/bin/sh
set -e
wget "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
unzip "liar_dataset.zip"
rm "liar_dataset.zip"
