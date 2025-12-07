import numpy as np 
import os
import pdb
import sys






def save_feats(feats_dir, dataset_file, labels_file):
    filenames = sorted(os.listdir(feats_dir))

    for filename in filenames:
        if filename.endswith("feats"):
            label    = np.array(int(filename.split("_")[0]), dtype="int32")
            filename = os.path.join(feats_dir, filename)

            print(filename)

            feat_file = open(filename, "rb")
            n_feats   = np.fromfile(feat_file, dtype="int32", count=1)
            feats     = np.fromfile(feat_file, dtype="float32")
            feat_file.close()

            feats.tofile(dataset_file)
            label.tofile(labels_file)





def main():
    if len(sys.argv) != 4:
        sys.exit("save_feats <dir_iftCNN_feats> <dataset.z> <labels.z>")

    feats_dir   = str(os.path.expanduser(sys.argv[1]))
    out_dataset = str(os.path.expanduser(sys.argv[2]))
    out_labels  = str(os.path.expanduser(sys.argv[3]))

    dataset_file = open(out_dataset, "wb")
    labels_file = open(out_labels, "wb")

    save_feats(feats_dir, dataset_file, labels_file)
    dataset_file.close()
    labels_file.close()



if __name__ == "__main__":
    sys.exit(main())

