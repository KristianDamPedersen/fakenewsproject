# Convert the csv to many parquet files, each containing 10000 rows
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

csv_file_path = 'cr_removed.csv'
parquet_file_prefix = 'dataset.parquet/file_'
parquet_file_suffix = '.parquet'

chunk_size = 10000  # number of rows per chunk

# Iterate over the CSV file in chunks and write each chunk to a separate Parquet file
for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunk_size)):
    # Create the filename for the current chunk
    ident = str(i).zfill(4)
    parquet_file_path = parquet_file_prefix + ident + parquet_file_suffix

    # Write the current chunk to a Parquet file
    table = pa.Table.from_pandas(chunk)
    pq.write_table(table, parquet_file_path)
    print("\rchunk", ident, end='')
print('operation finished')
