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
        sys.exit(f"python {sys.argv[0]} <specific_svoxel_dataset.zip> <tsne_perplexity> <output_plot_file.html>")

    symm_svoxels_dataset_path = sys.argv[1]
    perplexity = float(sys.argv[2])
    output_plot_file = sys.argv[3]
    parent_dir = os.path.dirname(output_plot_file)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print("------------------------------------------")
    print(f"- Specific Symmetric Supervoxel DataSet Path: {symm_svoxels_dataset_path}")
    print(f"- t-SNE perplexity: {perplexity}")
    print(f"- Output Plot File: {output_plot_file}")
    print("------------------------------------------\n")

    print('- Reading Datasets')
    Z = ift.ReadDataSet(symm_svoxels_dataset_path)
    X = Z.GetData()
    print(f"(n_samples, n_feats) = {X.shape}")

    print('- Projecting by tSNE')
    tsne = TSNE(n_components=2, verbose=1, n_iter=500, perplexity=perplexity, n_jobs=8)
    X_proj = tsne.fit_transform(X)

    trace_train = go.Scattergl(x=X_proj[:-1, 0], y=X_proj[:-1, 1], name='Training', mode='markers')
    trace_test = go.Scattergl(x=X_proj[-1:, 0], y=X_proj[-1:, 1], name='Testing', mode='markers')

    data = [trace_train, trace_test]

    symmetric_svoxels_feats_filename = os.path.basename(symm_svoxels_dataset_path).split(".zip")[0]
    layout= go.Layout(title= f'Specific Supervoxel Features from \'<i>{symmetric_svoxels_feats_filename}</i>\' - '
                             f'<b>t-SNE perplexity: {perplexity}</b>')

    fig = dict(data=data, layout=layout)
    ploff.plot(fig, filename=output_plot_file, auto_open=False)


if __name__ == "__main__":
    main()
