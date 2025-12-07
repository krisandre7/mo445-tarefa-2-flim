import numpy as np 
import os
import pdb
import sys



def save_splitted_set_idxs(result_dir, neg_idxs, pos_train_idxs, pos_test_idxs):
    """Save the idxs from the sets already splitted.
    """
    np.savetxt(os.path.join(result_dir, "neg_ids.txt"), neg_idxs, fmt="%d")
    np.savetxt(os.path.join(result_dir, "pos_train_ids.txt"), pos_train_idxs, fmt="%d")
    np.savetxt(os.path.join(result_dir, "pos_test_ids.txt"), pos_test_idxs, fmt="%d")


def write_time_stats(NM_proc_times, log):
    """Writes the NM processing times in the log.

    NM_proc_times: dict
        Dict with the processing times of the NM methods.

    log: file
        Log file.

    Notes
    -----
    e.g: NM_proc_times = 
    {
        0.05: {
            6.0: {
                'our_NM': [...],
                'random': [...]
            },
            8.0: {
                'our_NM': [...],
                'random': [...]
            }
        }
    }
    """
    log.write("###### TIME COMPUTATION #####\n")

    for neg_train_perc in sorted(NM_proc_times.keys()):
        for max_NM_time in sorted(NM_proc_times[neg_train_perc].keys()):
            for NM_method in sorted(NM_proc_times[neg_train_perc][max_NM_time].keys()):
                avg_time = np.mean(NM_proc_times[neg_train_perc][max_NM_time][NM_method])
                std_time = np.std(NM_proc_times[neg_train_perc][max_NM_time][NM_method])
                log.write("Neg. Train. Perc: %.2f%%\n" % (neg_train_perc*100))
                log.write("Max. NM Time: %.2f secs\n" % max_NM_time)
                log.write("NM: %s\n" % NM_method)
                log.write("Min. NM Proc. Time: %.2f secs\n" % np.min(NM_proc_times[neg_train_perc][max_NM_time][NM_method]))
                log.write("Max. NM Proc. Time: %.2f secs\n" % np.max(NM_proc_times[neg_train_perc][max_NM_time][NM_method]))
                log.write("Avg. NM Proc. Time: %.2f +- %.2f secs\n\n" % (avg_time, std_time))
    log.write("##########################\n")


def save_scores(output_dir, pos_scores, neg_scores):
    """Save the positive and negative scores into numpy files.
    pos_scores and neg_scores must have the same keys (THIS IS NOT VERIFIED).

    Parameters
    ----------
    output_dir: string
        Output Directory.

    pos_scores: dict
        Dictionary of scores from the positive sample classifications.

    neg_scores: dict
        Dictionary of scores from the negative sample classifications.

    Notes
    -----
    E.g. of the "shape" from the pos_scores and neg_scores
    {
        0.05: {
            6.0: {
                'our_NM': [...],
                'random': [...]
            },
            8.0: {
                'our_NM': [...],
                'random': [...]
            }
        }
    }
    """
    output_dir = os.path.expanduser(output_dir)
    for neg_train_perc in sorted(pos_scores.keys()):
        for max_NM_time in sorted(pos_scores[neg_train_perc].keys()):
            for NM_method in sorted(pos_scores[neg_train_perc][max_NM_time].keys()):
                pos_filename = os.path.join(output_dir, "pos.%s.time(%.2f).perc(%.2f).npy" % (NM_method, max_NM_time, neg_train_perc))
                neg_filename = os.path.join(output_dir, "neg.%s.time(%.2f).perc(%.2f).npy" % (NM_method, max_NM_time, neg_train_perc))

                np.save(pos_filename, pos_scores[neg_train_perc][max_NM_time][NM_method])
                np.save(neg_filename, neg_scores[neg_train_perc][max_NM_time][NM_method])






