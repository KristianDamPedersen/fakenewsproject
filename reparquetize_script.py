from dask.distributed import Client, LocalCluster
import dask.dataframe as dd

def main():
    # Start a local cluster
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print(client)

    for i in range(1, 10):
        print("Processing file: " + str(i))
        print("Reading file...")
        ddf = dd.read_parquet(f'./data/c{i}.parquet')
        print("Repartitioning...")
        ddf = ddf.repartition(npartitions=100)
        print("Writing file...")
        ddf.to_parquet(f'./data/new_parquets/c{i}.parquet')

if __name__ == '__main__':
    main()


