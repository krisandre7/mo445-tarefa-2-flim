import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage

def rotate_image(data):
    data = data.astype("int32")
    data = ndimage.interpolation.rotate(data, 180, axes=(0,1)) # 0 -- x axis; 1 -- y axis; 2 -- z axis;
    out = nib.analyze.AnalyzeImage(data, np.eye(4))
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit("python rotate_image.py <input_image> <output_rotate_image>")

    img_path = sys.argv[1]
    out_img_path = sys.argv[2]

    print(img_path)
    print(out_img_path)

    parent_dir = os.path.dirname(out_img_path)
    print("parent_dir = %s" % parent_dir)

    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    img = nib.load(img_path)
    out = rotate_image(img.get_data())
    nib.save(out, out_img_path)

    print("Done...\n")

if __name__ == "__main__":
    main()



