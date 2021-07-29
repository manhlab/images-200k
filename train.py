import pandas as pd
import numpy as np
import glob
import pickle
import os
import gc

import h5py
from tqdm import tqdm

query_filename = datasets[-1]

ids = []
embeddings = []
for dataset in tqdm(datasets):
    filename = f"features/{dataset}/data_features.pickle"
    if "query" in filename:
        query_filename = filename
        print(query_filename)
        continue
    file_to_read = open(filename, "rb")
    loaded_features = pickle.load(file_to_read)
    file_to_read.close()
    
    ids += loaded_features["ids"]
    embeddings += loaded_features["embeddings"]
    
print(len(ids), len(embeddings))
ids, embeddings = zip(*sorted(zip(ids, embeddings)))
print(ids[95763], embeddings[95763])

X = np.vstack(embeddings)
%%capture
if False:
    !pip install pca
    from pca import pca
    #nb_dim = 64 # max 256
    #model = pca(n_components=nb_dim, normalize=True)
    #results = model.fit_transform(X)
if False:
    from sklearn.decomposition import IncrementalPCA
    transformer = IncrementalPCA(n_components=256, batch_size=2000)
    transformer.partial_fit(X)
    X_transformed = transformer.fit_transform(X)
    X_transformed.shape

from sklearn import decomposition

n_comp = 256
svd = decomposition.TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd.fit(X)
print(svd.explained_variance_ratio_.sum())

X_transformed = svd.transform(X)
file_to_read = open(query_filename, "rb")
loaded_features = pickle.load(file_to_read)
file_to_read.close()

query_ids = loaded_features["ids"]
query_embeddings = loaded_features["embeddings"]

print(query_ids[0], query_embeddings[0])

query_ids, query_embeddings = zip(*sorted(zip(query_ids, query_embeddings)))
print(query_ids[38391], query_embeddings[38391])

query_X = np.vstack(query_embeddings)
query_X_transformed = svd.transform(query_X)
import h5py
import numpy as np

M_ref = X_transformed #np.random.rand(1_000_000, 256).astype('float32')
M_query = query_X_transformed #np.random.rand(50_000, 256).astype('float32')
print(M_ref.shape, M_query.shape)

qry_ids = ['Q' + str(x).zfill(5) for x in range(50_000)]
ref_ids = ['R' + str(x).zfill(6) for x in range(1_000_000)]

out = "fb-isc-submission.h5"
with h5py.File(out, "w") as f:
    f.create_dataset("query", data=M_query)
    f.create_dataset("reference", data=M_ref)
    f.create_dataset('query_ids', data=qry_ids)
    f.create_dataset('reference_ids', data=ref_ids)

