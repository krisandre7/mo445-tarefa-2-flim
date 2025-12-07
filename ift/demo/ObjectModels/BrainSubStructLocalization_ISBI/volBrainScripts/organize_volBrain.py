import glob
import os
import pdb
import shutil
import sys
import tempfile
import zipfile

import nibabel as nib
import numpy as np
from scipy import ndimage


def rotate_image(data):
    data = ndimage.interpolation.rotate(data, 180, axes=(0,1)) # 0 -- x axis; 1 -- y axis; 2 -- z axis;
    out = nib.analyze.AnalyzeImage(data, np.eye(4))
    return out


def convert_and_save(input_img_path, out_img_path):
    img = nib.load(input_img_path)    
    nib.save(img, out_img_path)



def convert_hippo_and_save(input_img_path, out_img_path):
    img = nib.load(input_img_path)

    data = img.get_data()
    data[data < 11] = 0
    data[data > 12] = 0
    data[data == 11] = 2 # left hippocampus
    data[data == 12] = 1 # right hippocampus

    out = nib.analyze.AnalyzeImage(data, np.eye(4))

    nib.save(out, out_img_path)


def convert_hemi_and_save(input_img_path, out_img_path):
    img = nib.load(input_img_path)

    data = img.get_data()
    relabel_data = np.array(data)
    relabel_data[data == 3] = 1 # left cerebellum --> cerebellum
    relabel_data[data == 4] = 1 # left cerebellum --> cerebellum

    relabel_data[data == 1] = 2 # left hemisphere --> right hemisphere
    relabel_data[data == 2] = 3 # right hemisphere --> left hemisphere

    relabel_data[data == 5] = 0 # brain stem --> bg

    out = nib.analyze.AnalyzeImage(relabel_data, np.eye(4))

    nib.save(out, out_img_path)



# The corresponding hippocampus labels, from the volBrain segmentation, are:
# Left Hippo = 11 will become to 2
# Right Hippo = 11 will become to 1
# Images are not in radiological orientation. We won't reorient them, because the ANTS deformation fields
# only works for the original orientation.
def main():
    if len(sys.argv) != 4:
        sys.exit("python organize_volBrain.py <input_dir> <out_dir> "
                 "<image space: native or mni>\n"
                 "-> input_dir = directory (control or pre-post) with the "
                 "image subdirectories. ex: 000001_000001_Pre, 000001_000002_Post, "
                 "etc\n"
                 "Ex: python organize_volBrain.py hippo_volBrain/controls "
                 "volBrain_organized/controls mni")

    input_dir = sys.argv[1]
    out_dir = sys.argv[2]
    img_space = sys.argv[3]

    if not os.path.exists(input_dir):
        sys.exit("Input Dir %s does not exist" % input_dir)

    if img_space != "mni" and img_space != "native":
        sys.exit("Invalid Image Space: %s... Try mni or native" % img_space)

    parent_dir = os.path.dirname(out_dir)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    out_orig_dir = os.path.join(out_dir, "orig")
    out_hippo_dir = os.path.join(out_dir, "labels/hippocampus")
    out_hemi_dir = os.path.join(out_dir, "labels/brain")

    if not os.path.exists(out_orig_dir):
        os.makedirs(out_orig_dir)
    if not os.path.exists(out_hippo_dir):
        os.makedirs(out_hippo_dir)
    if not os.path.exists(out_hemi_dir):
        os.makedirs(out_hemi_dir)

    if img_space == "mni":
        prefix = ""
    else:
        prefix = "native_"

    tmp_dir = tempfile.mkdtemp(prefix="volBrain_")

    for img_dir in os.listdir(input_dir):
        # example of file: 000001_000001.nii.gz_job45792220917222120.zip
        # example of file: native_000001_000001.nii.gz_job45792220917222120.zip
        zip_pattern = os.path.join(input_dir, img_dir, "%s%s*.zip" % (
            prefix, img_dir))

        zip_path = glob.glob(zip_pattern)[0]
        print("tmp_dir = %s" % tmp_dir)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        orig_img_pattern = os.path.join(tmp_dir, "%sn_mmni_fjob*nii" % prefix)
        hippo_img_pattern = os.path.join(tmp_dir, "%slab_n_mmni_fjob*nii" % prefix)
        hemi_img_pattern = os.path.join(tmp_dir, "%shemi_n_mmni_fjob*nii" % prefix)
        affine_file_pattern = os.path.join(tmp_dir, "affine_*")

        orig_img_path = glob.glob(orig_img_pattern)[0]
        hippo_img_path = glob.glob(hippo_img_pattern)[0]
        hemi_img_path = glob.glob(hemi_img_pattern)[0]


        print(zip_path)
        print(orig_img_path)
        print(hippo_img_path)
        print(hemi_img_path)


        out_orig_img_path = os.path.join(out_orig_dir, "%s.hdr" % img_dir)
        out_hippo_img_path = os.path.join(out_hippo_dir, "%s.hdr" % img_dir)
        out_hemi_img_path = os.path.join(out_hemi_dir, "%s.hdr" % img_dir)
        print(out_orig_img_path)
        print(out_hippo_img_path)
        print(out_hemi_img_path)
        print("")

        print("convert_and_save")
        convert_and_save(orig_img_path, out_orig_img_path)
        print("convert_hippo_and_save")
        convert_hippo_and_save(hippo_img_path, out_hippo_img_path)
        print("convert_hemi_and_save")
        convert_hemi_and_save(hemi_img_path, out_hemi_img_path)

        if img_space == "mni":
            print("affine_path")
            affine_path = glob.glob(affine_file_pattern)[0]
            out_affine_path = os.path.join(out_orig_dir, "%s_ANTS_DefFields_Affine.txt" % img_dir)
            shutil.copyfile(affine_path, out_affine_path)

        shutil.rmtree(tmp_dir)



if __name__ == "__main__":
    main()





