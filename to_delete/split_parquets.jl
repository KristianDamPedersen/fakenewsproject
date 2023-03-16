# REASON FOR DELETION: we already have a thing to split dataset into three with dask

using DataFrames, Parquet, Random

function split_df(df, train=0.8, val=0.1)
    rows = nrow(df)
    indices = shuffle(1:rows)

    train_size = Int(round(rows * train))
    val_size = Int(round(rows * val))

    test_size = rows - train_size - val_size

    train_indices = indices[begin:train_size]
    val_indices = indices[train_size+1:train_size+val_size]
    test_indices = indices[train_size+val_size+1:end]

    train_df = df[train_indices, :]
    val_df = df[val_indices, :]
    test_df = df[test_indices, :]
    df = []
    [train_df, val_df, test_df]
end

files = filter(x -> startswith(x, "chunk"), readdir())

dirs = ["train.parquet", "validation.parquet", "test.parquet"]

# Write the three datasets
for directory in dirs
    isdir(directory) || mkdir(directory)
end

for (i, file) in enumerate(files)
    @info "Opening file: $file"
    df = read_parquet(file) |> DataFrame |> disallowmissing
    for (j, dataset) in enumerate(split_df(df))
        @info "Writing dataset: $(dirs[j])"
        write_parquet(joinpath(dirs[j], "chunk_$i.parquet"), dataset)
    end
    df = []
end

