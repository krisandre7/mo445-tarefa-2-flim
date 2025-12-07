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
import plotly.io as plio

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
    overlapping_percs = np.array(Z.GetWeights())

    print('- Projecting by tSNE')
    tsne = TSNE(n_components=2, verbose=1, n_iter=500, perplexity=perplexity, n_jobs=8)
    X_proj = tsne.fit_transform(X)

    # colorbar = dict(
    #     title = 'Overlapping Percentage: Supervoxel and Lesion',
    #     titleside = 'top',
    #     tickmode = 'array',
    #     tickvals = [0, 1.0],
    #     ticktext = ['Normal Tissue', 'Lesion Tissue'],
    #     ticks = 'outside'
    # )

    point_opacity = 0.1
    # colorscale = [[0.0, f'rgb(165,0,38,{point_opacity})'], [0.1111111111111111, f'rgb(215,48,39,{point_opacity})'], [0.2222222222222222, f'rgb(244,109,67,{point_opacity})'],
    #             [0.3333333333333333, f'rgb(253,174,97,{point_opacity})'], [0.4444444444444444, f'rgb(254,224,144,{point_opacity})'], [0.5555555555555556, f'rgb(224,243,248,{point_opacity})'],
    #             [0.6666666666666666, f'rgb(171,217,233,{point_opacity})'],[0.7777777777777778, f'rgb(116,173,209,{point_opacity})'], [0.8888888888888888, f'rgb(69,117,180,{point_opacity})'],
    #             [1.0, f'rgb(49,54,149,{point_opacity})']]
    # colorscale = [, , 
    #             ,  
    #             ,, ,
    #             ]
    colorscale = [[0.0, f'rgb(49,54,149,{point_opacity})'], [0.1111111111111111, f'rgb(69,117,180,{point_opacity})'],
                  [0.2222222222222222, f'rgb(116,173,209,{point_opacity})'], [0.3333333333333333, f'rgb(171,217,233,{point_opacity})'],
                  [0.4444444444444444, f'rgb(224,243,248,{point_opacity})'], [0.5555555555555556, f'rgb(254,224,144,{point_opacity})'],
                  [0.6666666666666666, f'rgb(253,174,97,{point_opacity})'], [0.7777777777777778, f'rgb(244,109,67,{point_opacity})'],
                  [0.8888888888888888, f'rgb(215,48,39,{point_opacity})'], [1.0, f'rgb(165,0,38,{point_opacity})']]

    text = [f"Supervoxel {svoxel}" for svoxel in range(X.shape[0])]
    trace = go.Scattergl(x=X_proj[:, 0], y=X_proj[:, 1], mode='markers', text=text,
                         marker=dict(size=10, color=overlapping_percs, colorscale=colorscale, showscale=True,
                                     cmin=0, cmax=1.0, opacity=0.7,
                                     line=dict(
                                         color='rgb(0, 0, 0)',
                                         width=1.5
                                     ),
                                     colorbar=dict(
                                         title='',
                                         tick0=0,
                                         dtick=0.1,
                                         tickfont=dict(
                                             size=16
                                         )
                                     )))
    data = [trace]

    symmetric_svoxels_feats_filename = os.path.basename(symmetric_svoxels_feats_path).split(".zip")[0]
    layout= go.Layout(title= f'Supervoxel Histogram Features from \'<i>{symmetric_svoxels_feats_filename}</i>\' - '
                             f'<b>t-SNE perplexity: {perplexity}</b>',
                      xaxis=dict(
                          tickfont=dict(
                              size=16,
                              color='black'
                          )),
                      yaxis=dict(
                          tickfont=dict(
                              size=16,
                              color='black'
                          )))

    fig = dict(data=data, layout=layout)
    # plio.write_image(fig, output_plot_file.replace(".html", ".pdf"))
    ploff.plot(fig, filename=output_plot_file, auto_open=True)


if __name__ == "__main__":
    main()
