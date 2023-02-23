TEMPDIR=$(mktemp -d)
pushd $TEMPDIR

# Download all of the 9 parts including the main zip
for i in {01..09} "ip"
do
  wget "https://github.com/several27/FakeNewsCorpus/releases/download/v1.0/news.csv.z$i"
done

echo "Done downloading files:\n"

echo "Packing split archive into a single archive"
zip -FF "news.csv.zip" --out "joined_news.csv.zip"

popd

mv "$TEMPDIR/joined_news.csv.zip" .

rm -r $TEMPDIR
