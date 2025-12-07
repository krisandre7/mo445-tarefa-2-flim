import json
import numpy as np
import pdb
import os
import sys





def main():
    if len(sys.argv) != 2:
        sys.exit("build_mobio_dataset <configs.json>")

    json_file = open(sys.argv[1])
    try:
        configs = json.load(json_file)
    except (ValueError, KeyError, TypeError):
        Error.print_error("JSON format error", "main")

    dataset_file = open("mobio_feats.z", "wb")
    labels_file = open("mobio_labels.z", "wb")

    start_idx = 0
    end_idx   = 0
    print("- Reading Neg. samples")
    X_neg = np.fromfile(configs["FEATS"]["neg"], dtype="float32")
    y_neg = np.fromfile(configs["LABELS"]["neg"], dtype="int32")
    X_neg.tofile(dataset_file)
    y_neg.tofile(labels_file)

    end_idx += y_neg.size
    np.savetxt("mobio.neg_ids.txt", range(start_idx, end_idx), fmt="%d")
    del X_neg, y_neg


    start_idx = end_idx
    print("- Reading Male Gallery samples")
    X_male_gallery = np.fromfile(configs["FEATS"]["male_gallery"], dtype="float32")
    y_male_gallery = np.fromfile(configs["LABELS"]["male_gallery"], dtype="int32")
    X_male_gallery.tofile(dataset_file)
    y_male_gallery.tofile(labels_file)

    end_idx += y_male_gallery.size
    np.savetxt("mobio.male_gallery_ids.txt", range(start_idx, end_idx), fmt="%d")
    del X_male_gallery, y_male_gallery


    start_idx = end_idx
    print("- Reading Male Probe samples")
    X_male_probe = np.fromfile(configs["FEATS"]["male_probe"], dtype="float32")
    y_male_probe = np.fromfile(configs["LABELS"]["male_probe"], dtype="int32")
    X_male_probe.tofile(dataset_file)
    y_male_probe.tofile(labels_file)

    end_idx += y_male_probe.size
    np.savetxt("mobio.male_probe_ids.txt", range(start_idx, end_idx), fmt="%d")
    del X_male_probe, y_male_probe


    start_idx = end_idx
    print("- Reading Female Gallery samples")
    X_female_gallery = np.fromfile(configs["FEATS"]["female_gallery"], dtype="float32")
    y_female_gallery = np.fromfile(configs["LABELS"]["female_gallery"], dtype="int32")
    X_female_gallery.tofile(dataset_file)
    y_female_gallery.tofile(labels_file)

    end_idx += y_female_gallery.size
    np.savetxt("mobio.female_gallery_ids.txt", range(start_idx, end_idx), fmt="%d")
    del X_female_gallery, y_female_gallery


    start_idx = end_idx
    print("- Reading Female Probe samples")
    X_female_probe = np.fromfile(configs["FEATS"]["female_probe"], dtype="float32")
    y_female_probe = np.fromfile(configs["LABELS"]["female_probe"], dtype="int32")
    X_female_probe.tofile(dataset_file)
    y_female_probe.tofile(labels_file)

    end_idx += y_female_probe.size
    np.savetxt("mobio.female_probe_ids.txt", range(start_idx, end_idx), fmt="%d")
    del X_female_probe, y_female_probe
    
    dataset_file.close()
    labels_file.close()

if __name__ == "__main__":
    sys.exit(main())