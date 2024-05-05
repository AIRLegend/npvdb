import numpy as np
import logging
import sys
import time

from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger("vecdb")

if not logger.hasHandlers():
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("{asctime} {name} [{levelname}] {message}", style="{")
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)


def cosine_sim(m1, m2):

    m1 = m1.astype(np.float32)
    m2 = m2.astype(np.float32)
    magnitude_batch1 = np.linalg.norm(m1, axis=1, keepdims=True)
    magnitude_batch2 = np.linalg.norm(m2, axis=1, keepdims=True)
    scores = np.dot(m1, m2.T)
    mags = magnitude_batch1 * magnitude_batch2.T
    scores /= mags
    return scores


class VectorDB:

    def __init__(self, n_workers: int = 2):
        self._workers = dict()

        self.key2worker = dict()

        self.key2idx = dict()
        self.idx2key = dict()

        self._search_index = Index(n_centroids=n_workers)

        for i in range(n_workers):
            self._workers[i] = Worker()

    def init(self, keys, vecs):
        start = time.time()
        r = self.insert_batch(keys, vecs)
        logger.info(
            f"Loaded {len(keys)} vectors (dim={vecs.shape[-1]}) in {time.time()-start} secs."
        )
        return r

    def shard_data(self, keys, vecs, assignations):
        for i in np.unique(assignations):
            mask = assignations == i
            shrd = vecs[mask]
            keys_i = [k for k, i in zip(keys, mask) if i == True]
            self._workers[i].insert_batch(keys_i, shrd)
        return True

    def insert_batch(self, keys, vecs):
        if len(vecs.shape) != 2:
            raise ValueError("Only batches are allowed")

        if vecs.shape[0] != len(keys):
            raise ValueError("Lengths of vectors differ from keys")

        last_idx = len(self.key2idx)
        idxs = np.arange(len(keys)) + last_idx
        vec_dim = vecs.shape[-1]

        if (
            self._search_index.is_fitted and vec_dim != self._search_index.space_dim
        ):  # self.mat is not None and vec_dim != self.mat.shape[-1]:
            raise ValueError(
                "Passed vectors are not the same dim as the ones on this table."
            )

        for k in keys:
            if k in self.key2idx:
                raise ValueError(f"Key {k} already exists on the database")

        # Register entries
        for key, idx in zip(keys, idxs):
            self.key2idx[key] = idx
            # self.idx2key[idx] = key

        # Append vector to matrix
        if not self._search_index.is_fitted:
            self._search_index.fit(vecs)

        assignations = self._search_index.get_assignations(vecs)

        self.shard_data(keys, vecs, assignations)

        return True

    def get_closest(self, vec, k_nearest: int = 1):

        assignations = self._search_index.get_assignations(vec)

        unique_assigns = np.sort(np.unique(assignations))

        rets = []
        final_ret = []

        rets_assigns = dict()

        def _parallel_search(i):
            subvec = vec[assignations == i]
            if len(subvec.shape) == 1:
                subvec = subvec[None, :]
            return self._workers[i].get_closest(subvec, k_nearest=k_nearest), i

        # >80% of the computation is done here
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_parallel_search, i): i for i in unique_assigns}

        rets = [future.result() for future in as_completed(futures)]

        rets_assigns = {val[1]: i for i, val in enumerate(rets)}
        rets = [r[0] for r in rets]

        tmp_idxs = {i: 0 for i in unique_assigns}

        for assig_id in assignations:

            rets_subidx = rets_assigns[assig_id]
            final_ret.append(
                (rets[rets_subidx][0], rets[rets_subidx][1][tmp_idxs[assig_id]])
            )
            tmp_idxs[assig_id] += 1

        return final_ret

    def get_vector(self, key):
        raise RuntimeError("Not implemented!")

    def remove_vectors(self, key):
        raise RuntimeError("Not implemented!")


class Worker:

    def __init__(self):
        self.key2idx = dict()
        self.idx2key = dict()
        self.mat = None

    def insert_batch(self, keys, vecs):
        if len(vecs.shape) != 2:
            raise ValueError("Only batches are allowed")

        if vecs.shape[0] != len(keys):
            raise ValueError("Lengths of vectors differ from keys")

        last_idx = len(self.key2idx)
        idxs = np.arange(len(keys)) + last_idx
        vec_dim = vecs.shape[-1]

        if self.mat is None:
            self.mat = vecs
        else:
            self.mat = np.concatenate([self.mat, vecs])

        if vec_dim != self.mat.shape[-1]:
            raise RuntimeError(
                "Passed vectors are not the same dim as the ones on this table."
            )

        for key, idx in zip(keys, idxs):
            self.key2idx[key] = idx
            self.idx2key[idx] = key

    def get_closest(self, vec, k_nearest: int = 1, distance="cosine"):
        if distance != "cosine":
            raise ValueError("That distance parameter is not supported")

        if self.mat is None:
            raise RuntimeError("There are no vectors loaded!")

        if vec.shape[-1] != self.mat.shape[-1]:
            raise RuntimeError(f"Passed vector(s) has wrong dimensions {vec.shape}")

        # TODO: Make a function that abstracts the distance metric (eucliean, cosine, ...)
        scores = cosine_sim(self.mat, vec)

        ids = np.argsort(scores, axis=0)[-k_nearest:]
        entries = [[self.idx2key[i] for i in j] for j in ids.T]
        vecs = self.mat[ids].transpose(
            (1, 0, 2)
        )  # (v_1, closest vectors; v_2, closest_vecs; ...)

        return entries, vecs

    def get_vector(self, key):
        return self.mat[self.key2idx[key]]

    def remove_vectors(self, key):
        self.mat[self.key2idx[key]] = np.nan
        del self.key2idx[key]


class Index:

    def __init__(self, n_centroids):
        self._centers = None
        self._n_centroids = n_centroids

    def fit(self, data):
        km = KMeans(self._n_centroids, max_iter=50).fit(data)

        if self._centers is None:
            self._centers = np.array(km.cluster_centers_)

        return True

    def get_assignations(self, batch):
        assignations = np.linalg.norm(
            (self._centers[:, None, :] - batch),
            # (self._centers - batch),
            axis=-1,
        ).argmin(axis=0)

        return assignations

    @property
    def is_fitted(self):
        return not self._centers is None

    @property
    def space_dim(self):
        if not self.is_fitted:
            raise RuntimeError("Index is not initialized!")
        return len(self._centers)
