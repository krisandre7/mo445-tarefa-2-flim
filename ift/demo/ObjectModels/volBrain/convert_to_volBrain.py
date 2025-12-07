import csv
import os
import sys

import nibabel as nib
import numpy as np



def main():
    if len(sys.argv) != 3:
        sys.exit("usage: convert_to_volBrain image_hdr_set.csv output_dir")

    img_csv = sys.argv[1]
    out_dir   = sys.argv[2]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    with open(img_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        img_set = np.array(list(reader)).flatten().tolist()


    for img in img_set:
        out_img = os.path.join(out_dir, os.path.basename(img).replace(".hdr", ".nii.gz"))
        nib.save(nib.load(img), out_img)



if __name__ == "__main__":
    main()


