import os
import argparse
import json
from pathlib import Path
import shutil as sh

"""
Execution sample:

Parasites:

Separate model for each image:
python get_encoder.py parasites/entamoeba/images/ parasites/entamoeba/markers/ \
  parasites/arch2d_seg.json parasites/entamoeba/new_flim_sep

Merged model:
python get_encoder.py parasites/entamoeba/images/ parasites/entamoeba/markers/ \
  parasites/arch2d_seg.json parasites/entamoeba/new_flim_merged --merge

BraTS:

Separate model for each image:
python get_encoder.py brats/sample_images/ brats/markers/ \
  brats/arch3d_seg.json brats/new_flim_sep

Merged model:
python get_encoder.py brats/sample_images/ brats/markers/ \
  brats/arch3d_seg.json brats/new_flim_merged --merge
"""

def get_args():
    parser = argparse.ArgumentParser(
        prog="Sample to generate new-FLIM Encoder",
        description="Given a dataset, markers and a network architecture" \
                    " generates a FLIM model to an output dir"
    )
    parser.add_argument("dataset", help="Path to Dataset")
    parser.add_argument("markers", help="Path to Markers")
    parser.add_argument("architecture", help="Path to Architecture (json)")
    parser.add_argument("output_dir", help="Path to Save FLIM Outputs")
    parser.add_argument('--merge', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    return args

def get_arch(json_path):
  with open(json_path, "r") as file:
    arch_dict = json.load(file)

  return arch_dict["nlayers"], arch_dict

def extract_bag_of_features(images_folder, markers_folder, adj_radius,
                            background_fpoints, object_fpoints, bag_folder):
  """To generate the bag of feature points, we need the following arguments
    - images_folder: image folder;
    - markers_folder: marker folder;
    - adj_radius: adj. radius for patch extraction;
    - background_fpoitns: number of feature points on background markers;
    - object_fpoints: number of feature points on object markers;
    - bag_folder: new marker folder, representing a bag of feature points.
  """
  cmd = f"iftBagOfFeatPoints {images_folder} {markers_folder} {adj_radius}" \
          f" {background_fpoints} {object_fpoints} {bag_folder}"
  os.system(cmd)

def generate_encoder(n_layers, feature_points, arch_json, model_folder, merge):
  for input_layer in range(n_layers):
    """To generate new-FLIM layers, we need the following arguments:
      - feature_points: image folder;
      - arch_json: marker folder;
      - input_layer: layer number (1, 2, 3);
      - output_folder: folder to save encoder weights.
    """
    print(f"[INFO] Generating layer {input_layer+1} for each marked images")
    cmd = f"iftCreateLayerModel {feature_points} {arch_json} {input_layer+1}" \
            f" {model_folder}"
    os.system(cmd)
    if not merge:
      print(f"[INFO] Encoding layer {input_layer+1} for each marked images")
      """To generate the output of a new-FLIM layer, we need the following arguments:
        - arch_json: architecture of the network (.json)
        - input_layer: layer number (1, 2, 3)
        - model_folder: folder with the models
      """
      cmd = f"iftEncodeLayer {arch_json} {input_layer+1} {model_folder}"
      os.system(cmd)
    else:
      print(f"[INFO] Encoding and merging layer {input_layer+1}")
      """To merge new-FLIM layers, we need the following arguments:
        - arch_json: architecture of the network (.json)
        - input_layer: layer number (1, 2, 3)
        - model_folder: folder with the models
      """
      cmd = f"iftMergeLayerModels {arch_json} {input_layer+1} {model_folder}"
      os.system(cmd)

      """To encode merged new-FLIM layers, we need the following arguments:
        - arch_json: architecture of the network (.json)
        - input_layer: layer number (1, 2, 3)
        - model_folder: folder with the models
      """
      cmd = f"iftEncodeMergedLayer {arch_json} {input_layer+1} {model_folder}"
      os.system(cmd)

if __name__ == "__main__":
  args = get_args()

  n_layers, arch_dict = get_arch(args.architecture)

  os.makedirs(args.output_dir, exist_ok=True)

  # Extracts bag of features
  bofp_path = Path(args.output_dir) / Path("bofp")
  extract_bag_of_features(
    images_folder=args.dataset,
    markers_folder=args.markers,
    adj_radius=3,
    background_fpoints=2,
    object_fpoints=2,
    bag_folder=bofp_path
  )

  # Generates model for single image, or merged
  model_folder = Path(args.output_dir) / Path("model")
  # Verifies if model folder exists, and deletes it if so
  if os.path.exists(model_folder):
    sh.rmtree(model_folder)
  generate_encoder(
    n_layers=n_layers,
    feature_points=bofp_path,
    arch_json=args.architecture,
    model_folder=model_folder,
    merge=args.merge
  )

  # Moves layer folders to output folders
  layer_folders = [folder for folder in os.listdir(".") if "layer" in folder]
  for layer_folder in layer_folders:
    dst_layer_folder = Path(args.output_dir) / Path(layer_folder)
    # Verifies if layer folder exists, and deletes it if so
    if os.path.exists(dst_layer_folder):
      sh.rmtree(dst_layer_folder)
    sh.move(layer_folder, dst_layer_folder)