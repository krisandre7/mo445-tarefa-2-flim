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
from sklearn import svm
from sklearn import neighbors
import matplotlib.pyplot as plt
import opf_dataset as opf
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
    parser.add_argument('n_folds', type=int,
                        help='Number of folds for k-fold cross-validation (training set)')
    parser.add_argument('n_test_subsets', type=int,
                        help='Number of subsets to split the testing set')
    return parser

def print_args(args):
    print('--------------------------------------------')
    print('- Input OPF Dataset: %s' % args.opf_dataset)
    print('- Percentage to be used for training: %s' % args.perc_train)
    print('- Number of folds for k-fold cross-validation (training set): %s' % args.n_folds)
    print('- Number of subsets to split the testing set: %s' % args.n_test_subsets)
    print('--------------------------------------------\n')

def k_fold_cross_val(classif, X, Y, n_folds, classif_params):
    if classif in ['svm-linear', 'svm-rbf']:
        [cross_val_acc, cross_val_time] = k_fold_cross_val_svm(X, Y, n_folds, classif_params['kernel'])
    if classif in ['knn-uniform', 'knn-distance']:
        [cross_val_acc, cross_val_time] = k_fold_cross_val_knn(X, Y, n_folds, classif_params['weights'], classif_params['n_neighbors'])
    return [cross_val_acc, cross_val_time]

def k_fold_cross_val_svm(X, Y, n_folds, kernel):
    print('- Performing k-fold cross-validation with SVM-%s' % kernel)
    start = timeit.default_timer()
    k_fold = KFold(len(X), n_folds=n_folds, shuffle=True, random_state=0)
    clf = svm.SVC(kernel=kernel, decision_function_shape='ova')
    cross_val_acc = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1)
    cross_val_acc = np.mean(cross_val_acc)
    stop = timeit.default_timer()
    print("Time: %f" % (stop - start))
    return [cross_val_acc, (stop-start)]

def k_fold_cross_val_knn(X, Y, n_folds, weights, n_neighbors):
    print('- Performing k-fold cross-validation with kNN-%s' % weights)
    start = timeit.default_timer()
    k_fold = KFold(len(X), n_folds=n_folds, shuffle=True, random_state=0)
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    cross_val_acc = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1)
    cross_val_acc = np.mean(cross_val_acc)
    stop = timeit.default_timer()
    print("Time: %f" % (stop - start))
    return [cross_val_acc, (stop-start)]

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

def tsne_projection(X, dim):
    print('- Projecting by tSNE to %s dimensions' % dim)
    start = timeit.default_timer()
    tsne = TSNE(n_components=dim, verbose=1, perplexity=50, n_iter=2000, method='exact')
    X_proj = tsne.fit_transform(X)
    stop = timeit.default_timer()
    print("Time: %f" % (stop - start))
    return [X_proj, (stop-start)]

def split_dataset_into_array(X, n):
    X_array = list()
    last_t = 0
    for t in range(int(len(X)/n), len(X), int(len(X)/n)):
        X_array.append(X[last_t:t])
        last_t = t
    if last_t < len(X):
        X_array.append(X[last_t:])
    return X_array

def main():
    parser = build_argparse()
    args = parser.parse_args()

    if args.perc_train <= 0:
        print('Error: The training percentage must be greater than 0')
        sys.exit()

    if args.n_folds <= 0:
        print('Error: The number of folds for k-fold cross-validation must be at least 1')
        sys.exit()

    if args.n_test_subsets <= 0:
        print('Error: The number of test subsets must be at least 1')
        sys.exit()

    print_args(args)

    classif_list = ['svm-linear', 'svm-rbf', 'knn-uniform', 'knn-distance']
    classif_params = {'svm-linear':{'kernel':'linear'}, 'svm-rbf':{'kernel':'rbf'},
    'knn-uniform':{'weights':'uniform', 'n_neighbors':15}, 'knn-distance':{'weights':'distance', 'n_neighbors':15}}
    classif_colors = {'svm-linear':'red', 'svm-rbf':'green', 'knn-uniform':'blue', 'knn-distance':'cyan'}

    print('----------------------------------------------------------------------------------------')
    print('- Reading Dataset')
    print('----------------------------------------------------------------------------------------')
    Z = opf.read_dataset(args.opf_dataset)

    #split the dataset    
    # all_idxs = list(range(Z["feats"].shape[0]))
    # random.shuffle(all_idxs)
    # n_train = int(args.perc_train*len(all_idxs))
    
    # chosen_idxs = all_idxs[:n_train]
    # X_train = Z["feats"][chosen_idxs]
    # Y_train = Z["truelabel"][chosen_idxs]
    
    # chosen_idxs = all_idxs[n_train:]
    # X_test = Z["feats"][chosen_idxs]
    # Y_test = Z["truelabel"][chosen_idxs]

    sampler = StratifiedShuffleSplit(n_splits=1, test_size=(1.0-args.perc_train), random_state=0)
    X = Z["feats"]
    Y = Z["truelabel"]
    for train_index, test_index in sampler.split(X, Y):
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]

    print('Num. Samples = %s' % X.shape[0])
    print('Num. Features = %s' % X.shape[1])
    print('Num. Training samples = %s' % X_train.shape[0])
    print('Num. Testing samples = %s' % X_test.shape[0])

    #classification in the original dimension using SVM-linear and SVM-rbf
    orig_dim = X.shape[1]
    csv_list = list()
    csv_list.append(["dimension", "classifier", "cross_val_accuracy", "cross_val_time"])
    baseline_acc = dict()
    print('----------------------------------------------------------------------------------------')
    print('- Classification with SVM and kNN on %s dimensions (baseline)' % orig_dim)
    print('----------------------------------------------------------------------------------------')
    for classif in classif_list:
        [cross_val_acc, cross_val_time] = k_fold_cross_val(classif, X_train, Y_train, args.n_folds, classif_params[classif])
        csv_list.append([orig_dim, classif, cross_val_acc, cross_val_time])
        baseline_acc[classif] = cross_val_acc
        print('Accuracy (%s): %s' % (classif, cross_val_acc))

    with open("results_%s_baseline.csv" % (args.opf_dataset), "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter=';')
        writer.writerows(csv_list)

    #perform tSNE projections (in several dimensions) and then perform classifications
    print('----------------------------------------------------------------------------------------')
    print('- Performing tSNE projections and then classifications on those dimensions')
    dim = int(orig_dim / 2)
    best_dim_acc = {'svm-linear':0, 'svm-rbf':0, 'knn-uniform':0, 'knn-distance':0}
    best_dim = {'svm-linear':-1, 'svm-rbf':-1, 'knn-uniform':-1, 'knn-distance':-1}
    csv_list = list()
    csv_list.append(["dimension", "classifier", "cross_val_accuracy", "tsne_time", "cross_val_time"])
    x_plot = list()
    y_plot = {'svm-linear':list(), 'svm-rbf':list(), 'knn-uniform':list(), 'knn-distance':list()}
    while dim >= 2:
        print('----------------------------------------------------------------------------------------')
        #tSNE projection
        [X_train_proj, tsne_time] = tsne_projection(X_train, dim)

        #k-fold cross-validation with all the classifiers
        for classif in classif_list:
            [cross_val_acc, cross_val_time] = k_fold_cross_val(classif, X_train_proj, Y_train, args.n_folds, classif_params[classif])
            csv_list.append([dim, classif, cross_val_acc, tsne_time, cross_val_time])
            y_plot[classif].append(cross_val_acc)
            print('Accuracy (%s): %s' % (classif, cross_val_acc))

            if cross_val_acc >= best_dim_acc[classif]:
                best_dim_acc[classif] = cross_val_acc
                best_dim[classif] = dim

        x_plot.append(dim)
        dim = int(dim / 2)

    print('----------------------------------------------------------------------------------------')
    for classif in classif_list:
        print('Best dimension (%s): %s (acc=%s)' % (classif, best_dim[classif], best_dim_acc[classif]))
    
    with open("results_%s_training.csv" % (args.opf_dataset), "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter=';')
        writer.writerows(csv_list)

    #plot the results with matplotlib
    for classif in classif_list:
        line_baseline_acc = np.array([baseline_acc[classif] for i in x_plot])
        plt.semilogx(x_plot, line_baseline_acc, '--', color=classif_colors[classif], basex=2, label='%s-baseline'%(classif))
        plt.semilogx(x_plot, y_plot[classif], "o-", color=classif_colors[classif], basex=2, label="%s-training" % (classif))

    plt.xlabel('t-SNE dimension')
    plt.ylabel('cross-val accuracy')
    plt.legend(loc='lower right')
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.title('%s (%s feats)' % (args.opf_dataset, orig_dim))
    plt.grid(True)
    plt.savefig("plot_%s_training.png" % (args.opf_dataset))

    #split the testing set into several subsets
    print('----------------------------------------------------------------------------------------')
    print('- Splitting the testing set into %s subsets' % args.n_test_subsets)
    X_test_subsets = split_dataset_into_array(X_test, args.n_test_subsets)
    Y_test_subsets = split_dataset_into_array(Y_test, args.n_test_subsets)

    #perform tSNE projections and then classifications with the testing subsets
    if np.mean(list(best_dim.values())) == 0.0:
        print('- Performing tSNE projections (to %s dimensions) and then classifications on that dimension' % list(best_dim.values())[0])
        classif_acc_list = {'svm-linear':list(), 'svm-rbf':list(), 'knn-uniform':list(), 'knn-distance':list()}
        csv_list = list()
        csv_list.append(["test_subset", "classifier", "classif_accuracy", "tsne_time", "classif_time"])
        for t in range(len(X_test_subsets)):
            print('----------------------------------------------------------------------------------------')
            print('Testing subset: %s' % t)
            #tSNE projection
            X_join = np.concatenate((X_train, X_test_subsets[t]), axis=0)
            [X_join_proj, tsne_time] = tsne_projection(X_join, list(best_dim.values())[0])

            #classification
            X_train_proj = X_join_proj[:len(X_train)]
            X_test_proj = X_join_proj[len(X_train):]

            for classif in classif_list:
                [classif_acc, classif_time] = classification(classif, X_train_proj, Y_train, X_test_proj, Y_test_subsets[t], classif_params[classif])
                classif_acc_list[classif].append(classif_acc)
                csv_list.append([t, classif, classif_acc, tsne_time, classif_time])
                print('Accuracy (%s): %s' % (classif, classif_acc))

        print('----------------------------------------------------------------------------------------')
        print('Mean accuracy on the %s testing subsets:' % len(X_test_subsets))
        for classif in classif_list:
            print('%s: %s +- %s' % (classif, np.mean(classif_acc_list[classif]), np.std(classif_acc_list[classif])))

        with open("results_%s_testing.csv" % (args.opf_dataset), "w") as output:
            writer = csv.writer(output, lineterminator='\n', delimiter=';')
            writer.writerows(csv_list)

    #horiz_line_testing_acc = np.array([np.mean(classif_acc_list) for i in x_plot])
    #plt.semilogx(x_plot, horiz_line_acc_knn_distance, 'y--', basex=2, label="%s-testing" % (args.classif))

if __name__ == '__main__':
    main()


