import argparse
import os
import pdb
import sys
import random
import timeit

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import neighbors
import matplotlib.pyplot as plt
import opf_dataset as opf
import pyift as ift
import csv

def build_argparse():
    prog_desc = \
'''
Project a dataset in several dimensions by tSNE and trains the projections with the chosen classifier.
'''
    parser = argparse.ArgumentParser(description=prog_desc, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('opf_dataset', type=str,
                        help='OPF Dataset (*.zip)')
    parser.add_argument('perc_train', type=float,
                        help='Percentage to be used for training')
    parser.add_argument('pca_pre_proc', type=str,
                        help='Use PCA for pre-processing')
    return parser

def print_args(args):
    print('--------------------------------------------')
    print('- Input OPF Dataset: %s' % args.opf_dataset)
    print('- Percentage to be used for training: %s' % args.perc_train)
    print('- Use PCA for pre-processing: %s' % args.pca_pre_proc)
    print('--------------------------------------------\n')

def classification(classif, X_train, Y_train, X_test, Y_test, classif_params):
    if classif in ['svm-linear', 'svm-rbf']:
        [classif_acc, classif_time] = svm_classification(X_train, Y_train, X_test, Y_test, classif_params['kernel'])
    if classif in ['knn-uniform', 'knn-distance']:
        [classif_acc, classif_time] = knn_classification(X_train, Y_train, X_test, Y_test, classif_params['weights'], classif_params['n_neighbors'])
    return [classif_acc, classif_time]

def svm_classification(X_train, Y_train, X_test, Y_test, kernel):
    print('- Performing classification with SVM-%s' % kernel)
    start = timeit.default_timer()
    clf = svm.SVC(kernel=kernel, decision_function_shape='ova')
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
    stop = timeit.default_timer()
    print("Time: %f" % (stop - start))
    return [acc, (stop-start)]

def knn_classification(X_train, Y_train, X_test, Y_test, weights, n_neighbors):
    print('- Performing classification with kNN-%s' % weights)
    start = timeit.default_timer()
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
    stop = timeit.default_timer()
    print("Time: %f" % (stop - start))
    return [acc, (stop-start)]

def opf_classification(X_train, Y_train, X_test, Y_test):
    print('- Performing classification with OPF')
    start = timeit.default_timer()
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
    stop = timeit.default_timer()
    print("Time: %f" % (stop - start))
    return [acc, (stop-start)]

def tsne_projection(X, dim, pca_pre_proc):
    print('- Projecting by tSNE to %s dimensions' % dim)
    start = timeit.default_timer()
    tsne = TSNE(n_components=dim, verbose=1, perplexity=50, n_iter=2000, method='exact')

    if pca_pre_proc == 'yes':
        pca_dim = dim + int((X.shape[1]-dim)/3)
        print('Pre-processing data with PCA (to %s dimensions)' % (pca_dim))
        pca = PCA(n_components=pca_dim)
        X_pca = pca.fit_transform(X)
        X_proj = tsne.fit_transform(X_pca)
    else:
        X_proj = tsne.fit_transform(X)

    stop = timeit.default_timer()
    print("Time: %f" % (stop - start))
    return [X_proj, (stop-start)]

def main():
    parser = build_argparse()
    args = parser.parse_args()

    if args.perc_train <= 0:
        print('Error: The training percentage must be greater than 0')
        sys.exit()

    print_args(args)

    #some classifier params
    classif_list = ['svm-linear', 'svm-rbf', 'knn-uniform', 'knn-distance']
    classif_params = {'svm-linear':{'kernel':'linear'}, 'svm-rbf':{'kernel':'rbf'},
    'knn-uniform':{'weights':'uniform', 'n_neighbors':15}, 'knn-distance':{'weights':'distance', 'n_neighbors':15}}
    classif_colors = {'svm-linear':'red', 'svm-rbf':'green', 'knn-uniform':'blue', 'knn-distance':'cyan'}

    print('----------------------------------------------------------------------------------------')
    print('- Reading Dataset')
    print('----------------------------------------------------------------------------------------')
    Z = opf.read_dataset(args.opf_dataset)
    X, Y = Z["feats"], Z["truelabel"]

    if len(X) > 2000:
        print('Error: The dataset should have 2000 samples or less (due to t-SNE computational time)')
        sys.exit()

    #stratified sampling
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=(1.0-args.perc_train), random_state=0)
    for train_index, test_index in sampler.split(X, Y):
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]

    print('Num. Samples = %s' % X.shape[0])
    print('Num. Features = %s' % X.shape[1])
    print('Num. Training samples = %s' % X_train.shape[0])
    print('Num. Testing samples = %s' % X_test.shape[0])

    #perform tSNE projections (in several dimensions) and then perform classifications
    print('----------------------------------------------------------------------------------------')
    print('- Performing tSNE projections and then classifications on those dimensions')
    orig_dim = X.shape[1]
    dim = orig_dim
    best_dim_acc = {'svm-linear':0, 'svm-rbf':0, 'knn-uniform':0, 'knn-distance':0}
    best_dim = {'svm-linear':-1, 'svm-rbf':-1, 'knn-uniform':-1, 'knn-distance':-1}
    csv_list = list()
    csv_list.append(["dimension", "classifier", "classif_accuracy", "tsne_time", "classif_time"])
    x_plot = list()
    y_plot = {'svm-linear':list(), 'svm-rbf':list(), 'knn-uniform':list(), 'knn-distance':list()}
    baseline_acc = dict()

    #while dim >= 2:
    for dim in [orig_dim, 2]:
        print('----------------------------------------------------------------------------------------')
        #tSNE projection
        if dim != orig_dim:
            X_join = np.concatenate((X_train, X_test), axis=0)
            [X_join_proj, tsne_time] = tsne_projection(X_join, dim, args.pca_pre_proc)
            X_train_proj = X_join_proj[:len(X_train)]
            X_test_proj = X_join_proj[len(X_train):]

        #training/classification with all the classifiers
        for classif in classif_list:
            if dim == orig_dim:
                [classif_acc, classif_time] = classification(classif, X_train, Y_train, X_test, Y_test, classif_params[classif])
                baseline_acc[classif] = classif_acc
                tsne_time = 0

            else:
                [classif_acc, classif_time] = classification(classif, X_train_proj, Y_train, X_test_proj, Y_test, classif_params[classif])
                y_plot[classif].append(classif_acc)

            csv_list.append([dim, classif, classif_acc, tsne_time, classif_time])
            print('Accuracy (%s): %s' % (classif, classif_acc))

            if (dim != orig_dim) and (classif_acc >= best_dim_acc[classif]):
                best_dim_acc[classif] = classif_acc
                best_dim[classif] = dim

        if dim != orig_dim:
            x_plot.append(dim)
        #dim = int(dim / 2)

    print('----------------------------------------------------------------------------------------')
    for classif in classif_list:
        print('Best dimension (%s): %s (acc=%s)' % (classif, best_dim[classif], best_dim_acc[classif]))

    #save results in a CSV file
    with open("results_%s_train_test_without_cross_val.csv" % (args.opf_dataset), "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter=';')
        writer.writerows(csv_list)

    #plot the results with matplotlib
    for classif in classif_list:
        baseline_acc_line = np.array([baseline_acc[classif] for i in x_plot])
        plt.semilogx(x_plot, baseline_acc_line, '--', color=classif_colors[classif], basex=2, label='%s-baseline'%(classif))
        plt.semilogx(x_plot, y_plot[classif], "o-", color=classif_colors[classif], basex=2, label="%s-train-test" % (classif))

    plt.xlabel('t-SNE dimension')
    plt.ylabel('classif accuracy')
    plt.legend(loc='lower right')
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.title('%s (%s feats)' % (args.opf_dataset, orig_dim))
    plt.grid(True)
    plt.savefig("plot_%s_train_test_without_cross_val.png" % (args.opf_dataset))

if __name__ == '__main__':
    main()


