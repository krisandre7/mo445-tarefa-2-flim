import argparse
import os

import umap
import matplotlib.pyplot as plt

import pyift.pyift as ift


def build_argparse():
    prog_desc = \
        '''
Project a Supervoxel Dataset by UMAP.

The dataset file has n samples so that the first n-1 samples are training and
the last sample (index [n-1]) is a test sample.
.
'''

    parser = argparse.ArgumentParser(
        description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-d', '--supervoxel-dataset', type=str,
                        required=True, help='Supervoxel Dataset')

    parser.add_argument('-n', '--n-neighbors', type=int, default=15,
                        help='Number of Neighbors. Default: 15')
    parser.add_argument('-m', '--metric', type=str, default="euclidean",
                        help='Metric used to compute the distances in high dimensional space. '
                             'Options: [euclidean, manhattan, chebyshev, minkowski, canberra, '
                             'braycurtis, mahalanobis, wminkowski, seuclidean, cosine, '
                             'correlation, haversine, hamming, jaccard, dice, russelrao, kulsinski, '
                             'rogerstanimoto, sokalmichener, sokalsneath, yule]. Default: euclidean')

    return parser


def print_args(args):
    print('--------------------------------------------')
    print(f'- Supervoxel DataSet: {args.supervoxel_dataset}')
    print('--------------------------------------------')
    print(f'- Number of Neighbors: {args.n_neighbors}')
    print(f'- Metric: {args.metric}')
    print('--------------------------------------------\n\n')


def main():
    parser = build_argparse()
    args = parser.parse_args()
    print_args(args)

    print('- Reading Datasets')
    Z = ift.ReadDataSet(args.supervoxel_dataset)  # test control datasets

    X = Z.GetData()

    print('- Projecting by UMAP')
    umap_proj = umap.UMAP(n_neighbors=args.n_neighbors, metric=args.metric)
    X = umap_proj.fit_transform(X)

    Xtrain = X[:-1]
    Xtest = X[-1:]

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)
    ax.set_title('UMAP projection')
    line1, = ax.plot(Xtrain[:, 0], Xtrain[:, 1], 'ro',
                     linewidth=0.1, picker=5, label='Controls Train')
    line2, = ax.plot(Xtest[:, 0], Xtest[:, 1], 'bo',
                     linewidth=0.1, picker=5, label='Test')

    plt.legend(handles=[line1, line2])


    plt.show()



if __name__ == "__main__":
    main()
