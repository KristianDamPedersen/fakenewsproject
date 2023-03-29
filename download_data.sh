#!/bin/bash
# Download all of the 9 parts including the main zip
cd data/
for i in {01..09} "ip"
do
  wget "https://github.com/several27/FakeNewsCorpus/releases/download/v1.0/news.csv.z$i"
done

echo "Done downloading files:\n"

# Concat and extract
echo "Concatinating archives"
cat news.csv.{z01,z02,z03,z04,z05,z06,z07,z08,z09,zip} > temp.zip
rm news.csv.z*
echo "Extracting archive"
unzip temp.zip
rm temp.zip
# Remove carriage returns from articles.
sed 's/\r/ /g' news_cleaned_2018_02_13.csv > cr_removed.csv
rm news_cleaned_2018_02_13.csv
mkdir dataset.parquet
