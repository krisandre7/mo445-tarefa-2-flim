import os
from os.path import basename, join
from argparse import Namespace
import pyift as ift
from GPyOpt.methods import BayesianOptimization


def find_hyperparams(dir_img, dir_lab, dir_seeds, dir_out, dir_out_best, max_bopt_iter=15):
    """Find the best hyperparameters for LMNN using Bayesian Optimisation

    Parameters
    ----------

    dir_img : basestring
        Directory where the original images are located

    dir_lab : basestring
        Directory where the labels are located

    dir_seeds : basestring
        Directory where the image seeds are located

    dir_out : basestring
        Directory where the results are saved during optimization

    dir_out_best : basestring
        Directory where the best results are saved when the optimization finishes

    Returns
    -------
    dict:
        The best hyperparameters found.

    """

    images = sorted(os.listdir(dir_img))
    labels = sorted(os.listdir(dir_lab))
    seeds = sorted(os.listdir(dir_seeds))

    assert len(images) == len(labels) and len(images) == len(seeds)

    # Parameters boundaries

    args = Namespace()
    args.min_height = 0
    args.max_height = 10
    args.min_learn_rate = 1e-9
    args.max_learn_rate = 1e-3
    args.min_tree_size = 0
    args.max_tree_size = 1000
    args.min_n_targets = 5
    args.max_n_targets = 150
    args.min_n_impostors = 5
    args.max_n_impostors = 150

    bopt_iter = 0

    def optimize_clf(hyperparams):
        """ The actual objective function with packing and unpacking of hyperparameters

        Parameters
        ----------
        hyperparams : array_like
            Vector of hyperparameters to evaluate

        Returns
        -------
        float
            The validation error obtained.
        """

        hyperparams = hyperparams[0]
        height = int(round(hyperparams[0]))
        learn_rate = hyperparams[1]
        min_tree_size = int(round(hyperparams[2]))
        n_targets = int(round(hyperparams[3]))
        n_impostors = int(round(hyperparams[4]))

        nonlocal bopt_iter
        bopt_iter += 1
        print('Iteration {} of Bayesian Optimisation'.format(bopt_iter))
        print('Trying\n'
              'Height: {}\n'
              'Learning rate: {}\n'
              'Min. tree size: {}\n'
              '# targets: {}\n'
              '# impostors: {}\n'.format(height, learn_rate, min_tree_size, n_targets, n_impostors))

        A = ift.Circular(1.0)
        acc = 0.0

        for idx, img_path in enumerate(images):

            img  = ift.ReadImageByExt(join(dir_img, img_path))
            mimg = ift.ImageToMImage(img, ift.LABNorm_CSPACE)

            S   = ift.ReadSeeds(img, join(dir_seeds, seeds[idx]))
            lab = ift.Normalize(ift.ReadImageByExt(join(dir_lab, labels[idx])), 0, 1)

            segm = ift.LargeMarginDynTreeSegm(mimg, A, S, height, False, min_tree_size, n_targets,
                                              n_impostors, learn_rate, 10000)

            out = ift.Mask(img, segm)
            ift.WriteImageByExt(out, join(dir_out, basename(img_path)))

            acc += ift.DiceSimilarity(segm, lab)

        acc /= len(images)
        err = 1 - acc

        print('Evaluation error={:2.4f}\n'.format(err))
        return err


    domain = [{'name': 'height', 'type': 'discrete', 'domain': (args.min_height, args.max_height)},
              {'name': 'learn_rate', 'type': 'continuous', 'domain': (args.min_learn_rate, args.max_learn_rate)},
              {'name': 'min_tree_size', 'type': 'continuous', 'domain': (args.min_tree_size, args.max_tree_size)},
              {'name': 'n_targets', 'type': 'continuous', 'domain': (args.min_n_targets, args.max_n_targets)},
              {'name': 'n_impostors', 'type': 'continuous', 'domain': (args.min_n_impostors, args.max_n_impostors)}]

    bo = BayesianOptimization(f=optimize_clf, domain=domain, acquisition_type='LCB', num_cores=4)
    bo.run_optimization(max_iter=max_bopt_iter)

    solution = bo.x_opt
    print(solution)
    best_height = int(round(solution[0]))
    best_learn_rate = solution[1]
    best_tree_size = int(round(solution[2]))
    best_n_targets = int(round(solution[3]))
    best_n_impostors = int(round(solution[4]))

    A = ift.Circular(1.0)
    for idx, img_path in enumerate(images):
        img = ift.ReadImageByExt(join(dir_img, img_path))
        mimg = ift.ImageToMImage(img, ift.LABNorm_CSPACE)

        S = ift.ReadSeeds(img, join(dir_seeds, seeds[idx]))

        segm = ift.LargeMarginDynTreeSegm(mimg, A, S, best_height, False, best_tree_size, best_n_targets,
                                          best_n_impostors, best_learn_rate, 50000)

        out = ift.Mask(img, segm)
        ift.WriteImageByExt(out, join(dir_out_best, basename(img_path)))

    print('Best parameters\n'
          'Height: {}\n'
          'Learning rate: {}\n'
          'Min. tree size: {}\n'
          '# targets: {}\n'
          '# impostors: {}\n'.format(best_height, best_learn_rate, best_tree_size,
                                     best_n_targets, best_n_impostors))

    return {'height': best_height, 'learn_rate': best_learn_rate, 'tree_size': best_tree_size,
           'n_targets': best_n_targets, 'n_impostors': best_n_impostors}
