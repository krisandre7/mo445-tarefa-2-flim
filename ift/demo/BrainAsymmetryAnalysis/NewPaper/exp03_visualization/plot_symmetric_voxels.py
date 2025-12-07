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
    if len(sys.argv) != 3:
        sys.exit(f"python {sys.argv[0]} <symmetric_voxels_feats_2D.zip> <output_plot_file.html>")

    symmetric_voxels_feats_path = sys.argv[1]
    output_plot_file = sys.argv[2]
    parent_dir = os.path.dirname(output_plot_file)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print("------------------------------------------")
    print(f"- Symmetric Voxel Feats: {symmetric_voxels_feats_path}")
    print(f"- Output Plot File: {output_plot_file}")
    print("------------------------------------------\n")

    print('- Reading Datasets')
    Z = ift.ReadDataSet(symmetric_voxels_feats_path)
    X = Z.GetData()
    print(f"(n_samples, n_feats) = {X.shape}")
    labels = Z.GetTrueLabels()
    svoxels = Z.GetGroups()

    normal_voxels = np.where(labels == 0)[0]
    X_normal = X[normal_voxels]
    text = [f"Supervoxel {svoxels[voxel]}" for voxel in normal_voxels]
    trace_normal = go.Scattergl(x=X_normal[:, 0], y=X_normal[:, 1], name='Normal', mode='markers', text=text)
    print(f"X_normal.shape: {X_normal.shape}")

    lesion_svoxels = np.where(labels != 0)[0]
    X_lesion = X[lesion_svoxels]
    text = [f"Supervoxel {svoxels[voxel]}" for voxel in lesion_svoxels]
    trace_lesion = go.Scattergl(x=X_lesion[:, 0], y=X_lesion[:, 1], name='Lesion', mode='markers', text=text)
    print(f"X_lesion.shape: {X_lesion.shape}")

    data = [trace_normal, trace_lesion]

    symmetric_voxels_feats_filename = os.path.basename(symmetric_voxels_feats_path).split(".zip")[0]
    layout= go.Layout(title= f'Voxel Features from \'<i>{symmetric_voxels_feats_filename}</i>\'')

    fig = dict(data=data, layout=layout)
    ploff.plot(fig, filename=output_plot_file, auto_open=False)


if __name__ == "__main__":
    main()
