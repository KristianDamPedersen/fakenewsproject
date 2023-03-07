using Parquet, DataFrames, Missings

const df_template = DataFrame(
    id=Int[],
    content=String[],
    type=String[],
    title=String[],
    authors=String[],
    domain=String[],
    url=String[]
)

function process_file(file_name)
    new_df = read_parquet(file_name) |> DataFrame
    select!(new_df, [:id, :content, :type, :title, :authors, :domain, :url])
    for col in names(new_df)[2:end]
        new_df[!, col] = convert.(String, disallowmissing(replace(new_df[!, col], missing => "")))
    end

    new_df[!, :id] = convert.(Int, disallowmissing(replace(new_df[!, :id], missing => -1)))

    new_df
end


# Filter all files and only get the ones ending in .parquet
files = filter(x -> endswith(x, ".parquet"), readdir())
filter!(x -> x ≠ "dataset.parquet", files)

# Max rough size of each parquet in bytes
parquet_size = 2e9

path = "dataset.parquet"
global i = 1
while true
    # Initialise
    size = 0
    df = df_template
    while size < parquet_size
        size += filesize(files[i])
        @info "Reading $i of $(length(files))"
        df = vcat(df, process_file(files[i]))
        @info "Eltypes:" eltype.(eachcol(df))
        if i==length(files)
            break
        end
        global i += 1
    end
    write_parquet(joinpath(path, "chunk_$(i-1).parquet"), df)
    @info "Wrote chunk"
        
end
