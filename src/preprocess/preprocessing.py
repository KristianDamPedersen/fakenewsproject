# This is step 1
import pandas as pd
import random
random.seed(42)

file_name = "shuffled_deduped.parquet"
df = pd.read_parquet(file_name, columns=["type", "content"])
# Filter out the 'empty' and 'unknown' types
df = df[df['type'].isin(['', 'unknown']) == False]
df_size = df.shape[0]

n_non_reliable = 3897597
n_reliable =     1808242
downsample_factor = n_reliable / n_non_reliable


train_size = round(df_size * 0.8)
val_size = (df_size - train_size) // 2
test_size = df_size - train_size - val_size

def dicethrow(thistype):
    if thistype == 'reliable':
        return True
    rand = random.random()
    if rand > downsample_factor:
        return False
    return True

# Generate y values for models.
df["class"] = df["type"] == "reliable"

train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:train_size+val_size]
test_data = df.iloc[train_size + val_size:]

train_data = train_data[train_data['type'].map(dicethrow)]

# WRITE PARQUETS
def parquetise(df, path:str, set_type:str, chunk_size = 100000):
    """Parquetise some dataframe
    input: df (dataframe),
    path: The relative path, such as 'many_parquet_files_here.parquet/'
    set_type, string. typically "train", "val" or "test",
    chunk_size (optional): the number of rows per chunk"""

    file_n = 0
    df_length = len(df)
    for i in range(0, df_length, chunk_size):
        ident = str(file_n).zfill(6)

        chunk_df = df.iloc[i : min(i + chunk_size, df_length)]
        filename = f"{set_type}_{ident}.parquet"
        print("Writing:", filename, '\r', end='')
        chunk_df.to_parquet(path+filename, index=False)
        file_n += 1
    print()
    print('done!')
    
parquetise(train_data, "train.parquet/", "train")
parquetise(val_data, "val.parquet/", "val")
parquetise(test_data, "test.parquet/", "test")
# DELETE BELOW?

# Set the number of samples for 'reliable' and other categories
#num_samples = 1808242


# Group the DataFrame by the 'type' column
#df = df.groupby('type')


### Define a function to sample rows from each group
#def sample_rows(group):
#    if group.name == 'reliable':
#        return group.sample(min(len(group), num_samples))
#    else:
#        remaining_samples = num_samples - len(df[df['type'] == 'reliable'])
#        return group.sample(min(len(group), remaining_samples), replace=True)
#
## Apply the function to each group of rows in the grouped DataFrame
#sampled_groups = grouped_df.apply(sample_rows()))
#
## Concatenate the sampled groups back into a single DataFrame
#sampled_df = pd.concat(sampled_groups)
#
## Reset the index
#sampled_df.reset_index(drop=True, inplace=True)
#
#print(sampled_df)

