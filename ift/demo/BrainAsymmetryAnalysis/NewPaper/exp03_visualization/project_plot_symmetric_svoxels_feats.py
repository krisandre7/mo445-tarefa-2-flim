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
        sys.exit(f"python {sys.argv[0]} <symmetric_svoxels_feats.zip> <tsne_perplexity> <output_plot_file.html>")

    symmetric_svoxels_feats_path = sys.argv[1]
    perplexity = float(sys.argv[2])
    output_plot_file = sys.argv[3]
    parent_dir = os.path.dirname(output_plot_file)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print("------------------------------------------")
    print(f"- Symmetric Supervoxel Histogram Feats DataSet Path: {symmetric_svoxels_feats_path}")
    print(f"- t-SNE perplexity: {perplexity}")
    print(f"- Output Plot File: {output_plot_file}")
    print("------------------------------------------\n")

    print('- Reading Datasets')
    Z = ift.ReadDataSet(symmetric_svoxels_feats_path)
    X = Z.GetData()
    print(f"(n_samples, n_feats) = {X.shape}")
    labels = Z.GetTrueLabels()

    print('- Projecting by tSNE')
    tsne = TSNE(n_components=2, verbose=1, n_iter=500, perplexity=perplexity, n_jobs=8)
    X_proj = tsne.fit_transform(X)

    normal_svoxels = np.where(labels == 0)[0]
    X_proj_normal = X_proj[normal_svoxels]
    text = [f"Supervoxel {svoxel}" for svoxel in normal_svoxels]
    trace_normal = go.Scattergl(x=X_proj_normal[:, 0], y=X_proj_normal[:, 1], name='Normal', mode='markers', text=text)

    lesion_svoxels = np.where(labels != 0)[0]
    X_proj_lesion = X_proj[lesion_svoxels]
    text = [f"Supervoxel {svoxel}" for svoxel in lesion_svoxels]
    trace_lesion = go.Scattergl(x=X_proj_lesion[:, 0], y=X_proj_lesion[:, 1], name='Lesion', mode='markers', text=text)
    data = [trace_normal, trace_lesion]

    symmetric_svoxels_feats_filename = os.path.basename(symmetric_svoxels_feats_path).split(".zip")[0]
    layout= go.Layout(title= f'Supervoxel Histogram Features from \'<i>{symmetric_svoxels_feats_filename}</i>\' - '
                             f'<b>t-SNE perplexity: {perplexity}</b>')

    fig = dict(data=data, layout=layout)
    ploff.plot(fig, filename=output_plot_file, auto_open=False)


if __name__ == "__main__":
    main()
