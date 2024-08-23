# much of this code is based off of these two tutorials:
#   - https://github.com/facebookresearch/faiss/wiki/Getting-started
#   - https://www.pinecone.io/learn/series/faiss/hnsw/

import faiss
import numpy as np

# Vector Database Params
d = 128 # dimensionality
M = 32 # max number of friends
nb = 100000 # number of vectors in train set
nq = 1000 # number of vectors in query/test set
efConstruction = 40 #  how many entry points will be explored between layers during the construction
efSearch = 16 # how many entry points will be explored between layers during the search

# generating a dataset
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arrange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

index = faiss.IndexHNSWFlat(d, M)
# print(index.hnsw)

index.add(xb)
index.hnsw.max_level # prints the max level
index.hnsw.entry_point # prints the entry point

index.hnsw.efConstruction = efConstruction
index.hnsw.efSearch = efSearch


