from pprint import pprint
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

import plotly.offline as ploff
import plotly.graph_objs as go

import pyift.pyift as ift


def main():
    if len(sys.argv) != 4:
        sys.exit(f"python {sys.argv[0]} <feats.zip> <tsne_perplexity> <out_projected_feats.zip>")

    feats_path = sys.argv[1]
    perplexity = float(sys.argv[2])
    output_feats_path = sys.argv[3]
    parent_dir = os.path.dirname(output_feats_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print("------------------------------------------")
    print(f"- Feats: {feats_path}")
    print(f"- t-SNE perplexity: {perplexity}")
    print(f"- Output Projected: {output_feats_path}")
    print("------------------------------------------\n")

    print('- Reading Datasets')
    Z = ift.ReadDataSet(feats_path)
    X = Z.GetData()
    print(f"(n_samples, n_feats) = {X.shape}")
    true_labels = Z.GetTrueLabels()
    groups = Z.GetGroups()

    print('- Projecting by tSNE')
    tsne = TSNE(n_components=2, n_iter=500, verbose=1, perplexity=perplexity, n_jobs=8)
    Xproj = tsne.fit_transform(X)
    Xproj = Xproj.astype("float32")
    print(f"(n_samples, n_feats) = {Xproj.shape}")

    Zproj = ift.CreateDataSetFromNumPy(Xproj, true_labels)
    Zproj.SetGroups(groups)

    ift.WriteDataSet(Zproj, output_feats_path)


if __name__ == "__main__":
    main()
