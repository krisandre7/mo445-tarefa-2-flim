import glob
import os
import pdb
import shutil
import sys
import tempfile
import zipfile

import numpy as np
from scipy import ndimage

import pyift as ift



def convert_and_save(input_img_path, out_img_path):
    ift.WriteImageByExt(ift.ReadImageByExt(input_img_path), out_img_path)


def convert_hippo_and_save(input_img_path, out_img_path):
    img = ift.ReadImageByExt(input_img_path)
    data = img.AsNumPy()
    voxel_sizes = img.GetVoxelSizes()

    data[data < 11] = 0
    data[data > 12] = 0
    data[data == 11] = 2 # left hippocampus
    data[data == 12] = 1 # right hippocampus

    out = ift.CreateImageFromNumPy(data, True)
    out.SetVoxelSizes(voxel_sizes)

    ift.WriteImageByExt(out, out_img_path)


def convert_hemi_and_save(input_img_path, out_img_path):
    img = ift.ReadImageByExt(input_img_path)
    data = img.AsNumPy()
    voxel_sizes = img.GetVoxelSizes()

    relabel_data = np.array(data)
    relabel_data[data == 1] = 10 # right hemisphere

    relabel_data[data == 3] = 1 # left cerebellum --> cerebellum
    relabel_data[data == 4] = 1 # left cerebellum --> cerebellum
    relabel_data[data == 5] = 0 # brain stem --> bg

    relabel_data[relabel_data == 10] = 3 # right hemisphere

    out = ift.CreateImageFromNumPy(relabel_data, True)
    out.SetVoxelSizes(voxel_sizes)

    ift.WriteImageByExt(out, out_img_path)



# The corresponding hippocampus labels, from the volBrain segmentation, are:
# Left Hippo = 11 will become to 2
# Right Hippo = 11 will become to 1
# Images are not in radiological orientation. We won't reorient them, because the ANTS deformation fields
# only works for the original orientation.
def main():
    if len(sys.argv) != 4:
        sys.exit("python organize_volBrain.py <input_dir> <out_dir> "
                 "<image space: native or mni>\n"
                 "-> input_dir = directory all volBrain files (zip and pdf)")

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
        prefix_regex = "[!native_]"
    else:
        prefix = "native_"
        prefix_regex = "native_"


    tmp_dir = tempfile.mkdtemp(prefix="volBrain_")

    for img_zip in glob.glob(os.path.join(input_dir, "%s*.zip") % prefix_regex):
        # example of file: 000001_000001.nii.gz_job45792220917222120.zip
        # example of file: native_000001_000001.nii.gz_job45792220917222120.zip

        with zipfile.ZipFile(img_zip, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        orig_img_pattern = os.path.join(tmp_dir, "%sn_mmni_fjob*nii" % prefix)
        hippo_img_pattern = os.path.join(tmp_dir, "%slab_n_mmni_fjob*nii" % prefix)
        hemi_img_pattern = os.path.join(tmp_dir, "%shemi_n_mmni_fjob*nii" % prefix)
        affine_file_pattern = os.path.join(tmp_dir, "affine_*")

        orig_img_path = glob.glob(orig_img_pattern)[0]
        hippo_img_path = glob.glob(hippo_img_pattern)[0]
        hemi_img_path = glob.glob(hemi_img_pattern)[0]


        print(img_zip)
        print(orig_img_path)
        print(hippo_img_path)
        print(hemi_img_path)

        base = img_zip.split(".nii")[0]
        if prefix: # '' does not work for splitting
            base = base.split(prefix)[1]

        out_orig_img_path = os.path.join(out_orig_dir, "%s.nii.gz" % base)
        out_hippo_img_path = os.path.join(out_hippo_dir, "%s.nii.gz" % base)
        out_hemi_img_path = os.path.join(out_hemi_dir, "%s.nii.gz" % base)
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





