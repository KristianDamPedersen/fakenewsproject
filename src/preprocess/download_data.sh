#!/bin/bash

TEMPDIR=$(mktemp -d)

pushd $TEMPDIR
# Download all of the 9 parts including the main zip
for i in {01..09} "ip"
do
  wget "https://github.com/several27/FakeNewsCorpus/releases/download/v1.0/news.csv.z$i"
done
popd

echo "Done downloading files:\n"

echo "Extracting archives"
7z x $TEMPDIR/news.csv.zip -y -o"../../data" && rm -r $TEMPDIR
