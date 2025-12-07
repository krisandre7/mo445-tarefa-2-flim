import argparse
import os
import shutil as sh
from pathlib import Path

"""
Execution sample:

Parasites:
python decode.py parasites/arch2d_seg.json parasites/entamoeba/new_flim_merged/ 1

BraTS:
python decode.py brats/arch3d_seg.json brats/new_flim_sep/ 1
"""

def decode(layer, arch_json, model_folder, output_folder):
  cmd = f"iftDecodeLayer {layer} {arch_json} {model_folder} {output_folder}"
  os.system(cmd)

def get_args():
    parser = argparse.ArgumentParser(
        prog="Sample to decode using a new-FLIM Model",
        description="Given a new-FLIM model, decode the specified layer into a saliency map"
    )
    parser.add_argument("architecture", help="Path to Architecture (json)")
    parser.add_argument("flim_folder", help="Path to new-FLIM data folder")
    parser.add_argument("layer_n", help="Layer number (1, 2, ...)")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
  args = get_args()

  # Copies layer folders to current dir (.) [IFT Requirement]
  layer_folders = [
    folder for folder in os.listdir(args.flim_folder) if "layer" in folder
  ]
  for layer_folder in layer_folders:
    sh.copytree(Path(args.flim_folder) / Path(layer_folder), f"./{layer_folder}")

  decode(
    layer=args.layer_n,
    arch_json=args.architecture,
    model_folder=Path(args.flim_folder) / Path("model"),
    output_folder=Path(args.flim_folder) / Path("saliency")
  )

  # Removes layer subfolder from current dir()
  for layer_folder in layer_folders:
    sh.rmtree(layer_folder)