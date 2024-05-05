# npvDB - numpy Vector DB

Inspired by this tweet from the almighty Karpathy I decided to give it a try:

<html>
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">np.array<br>people keep reaching for much fancier things way too fast these days <a href="https://twitter.com/karpathy/status/1647374645316968449?ref_src=twsrc%5Etfw"></p>&mdash; Andrej Karpathy (@karpathy) April 15, 2023</a></blockquote> 
</html>

This is a (toy) vector database implemented almost entirely on Numpy and vanilla Python

## Install

```bash
git clone https://github.com/AIRLegend/npvdb
cd npvdb
pip install .
```

## Usage

```python
# Load dataset as a Numpy matrix
# keys = list with strings
# data = embeddings matrix
keys, data = load_data()

db = VectorDB(n_workers = 3 * int(np.log(len(keys))))

db.init(
    keys,
    data
)

# Get top 3 closest vectors
closest = db.get_closest(
    data[:1],
    k_nearest=3
)
```

## Structure

- There exist `Worker`s that store a separate shard of the embeddings matrix. Each worker runs INSERTs and SELECTs only within their shard (partition of the vector space).
- The space is partitioned in a Voronoi scheme using KMeans. The component resonsible for doing this is the `Index`. Once trained, stores the centroids.
- The database has a "master" (`VectorDB`) process which is in charged of orchestrating the sharding, insertion of new data and read of existing one.


## Where to get testing data
You can either use a `np.random.uniform` matrix for testing or load your own dataset. It should work as long as it is a `np.array`.

Some sources are benchmarks datasets. [Here you can access several of them](https://github.com/erikbern/ann-benchmarks/).

**Download NYC embeddings benchmark dataset**

```
wget http://ann-benchmarks.com/nytimes-256-angular.hdf5
```

Load them with

```python
import h5py

with h5py.File('nytimes-256-angular.hdf5', 'r') as file:
    # List all groups and datasets within the file
    print("Groups and Datasets in the HDF5 file:")
    print(list(file.keys()))

    data = file['train'][:]   # also test

    # Print the shape and datatype of the data
    print("Shape of the data:", data.shape)
    print("Datatype of the data:", data.dtype)
```




## TODO Ideas
- [ ] Add Save & load functions so the database and internal index can be persisted between sessions.
- [ ] Add tests.
- [ ] Map each worker to separate processes and avoid using `ThreadPoolExecutor`.
- [ ] Add suport for new index systems.
    - right now there is only one index supported (KMeans). The ability to build new ones based on other techniques such as LSH, Hierarchical, or random projections.

- [ ] Add suport for new search distance metrics (RN only `cosine` is supported).

- [ ] Add support for iterative initialization in case you cannot load `data` in a numpy array. Calling `init` with a generator loading parts of the vector matrix would solve this issue.

- [ ] Dynamic initialization of centroids:
    - Right now the user must specify the number of space partitions via `n_worker` parameter. This presents a tradeoff between search speed, memory consumption and query latency. 
    - One idea would be to compute several KMeans centroids on each worker and then aggregate them on the "master" process to update the `_search_index`.

- [ ] Add network support for horizontal scaling. Workers could communicate over TCP.


