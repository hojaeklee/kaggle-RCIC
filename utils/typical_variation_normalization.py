import numpy as np
from sklearn.decomposition import PCA
import torch

def compute_TVN_transform(negative_embedding):
    """
    Compute transformation for TVN (Typical Variation Normalization) based on https://pdfs.semanticscholar.org/463b/1c43dfd689841d2fd43d24058e68b7224dcf.pdf
    """
    neg_emb = negative_embeddings.numpy()
    pca = PCA() # keep all components
    pca.fit(neg_emb)

    axes = pca.components_ # (n_components, n_features)
    axes = torch.from_numpy(axes)
    mean = axes.mean(dim = 0)
    std = axes.std(dim = 0)

    return mean, std

def TVN_transform(experiment_embedding, negative_embedding):
    mean, var = compute_TVN_transform(negative_embedding)
    transformed_embedding = (experiment_embedding - mean) / var

    return transformed_embedding

### Script ###
import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd



    

