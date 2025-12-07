import numpy as np 
import os
import pdb
import sys






def save_feats(sample_filenames, dataset_file, labels_file):
    for filename in sample_filenames:
        filename = filename.split("\n")[0]
        if filename.endswith("feats"):
            label = np.array(int(filename.split("/")[-1].split("_")[0]), dtype="int32")

            print(filename)

            feat_file = open(filename, "rb")
            n_feats   = np.fromfile(feat_file, dtype="int32", count=1)
            feats     = np.fromfile(feat_file, dtype="float32")
            feat_file.close()

            feats.tofile(dataset_file)
            label.tofile(labels_file)





def main():
    """Converts the dataset in .feats to an only file to be read in numpy.
       The inputs are one protocol file from mobio/Mobio_IFT_Protocols/CNN
    """
    if len(sys.argv) != 4:
        sys.exit("save_Mobio_feats <protocol_file> <out_dataset.z> <out_labels.z>")

    sample_filenames = open(str(os.path.expanduser(sys.argv[1])), "r").readlines()
    out_dataset      = str(os.path.expanduser(sys.argv[2]))
    out_labels       = str(os.path.expanduser(sys.argv[3]))

    dataset_file = open(out_dataset, "wb")
    labels_file  = open(out_labels, "wb")
    save_feats(sample_filenames, dataset_file, labels_file)
    dataset_file.close()
    labels_file.close()



if __name__ == "__main__":
    sys.exit(main())

