import os
import argparse
import json
from pathlib import Path
import shutil as sh

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

def generate_sorted_encoder(
    n_layers, feature_points, arch_json, model_folder, merge, bg_filters,
    fg_filters
):
  for idx, input_layer in enumerate(range(n_layers)):
    """To generate new-FLIM layers, we need the following arguments:
      - feature_points: image folder;
      - arch_json: marker folder;
      - input_layer: layer number (1, 2, 3);
      - output_folder: folder to save encoder weights.
      - bg_filters: N filters for background on layer.
      - fg_filters: N filters for foreground on layer
    """
    print(f"[INFO] Generating layer {input_layer+1} for each marked images")
    cmd = f"iftCreateSortedLayerModel {feature_points} {arch_json} {input_layer+1}" \
            f" {model_folder} {bg_filters[idx]} {fg_filters[idx]}"
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

def sort_bag_of_features(
    images_folder, bag_folder, sorted_bag_folder, arch
):
    """To sort a bag of feature points folder:
        - images_folder: image folder
        - bag_folder: Bag Of Feature Points Folder
        - sorted_bag_folder: OutPut Folder (For Sorted Feature Points)
        - arch: Path to Json Network Architecture
    """
    cmd = f"iftSortBagOfFeatPoints {images_folder} {bag_folder} " \
            f"{sorted_bag_folder} {arch}"
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
  sorted_bofp_path = Path(args.output_dir) / Path("bofp_sorted")
  sort_bag_of_features(
    images_folder=args.dataset,
    bag_folder=bofp_path,
    sorted_bag_folder=sorted_bofp_path,
    arch=args.architecture
  )

  # Generates model for single image, or merged
  model_folder = Path(args.output_dir) / Path("model")
  # Verifies if model folder exists, and deletes it if so
  if os.path.exists(model_folder):
    sh.rmtree(model_folder)
  generate_sorted_encoder(
    n_layers=n_layers,
    feature_points=sorted_bofp_path,
    arch_json=args.architecture,
    model_folder=model_folder,
    merge=args.merge,
    bg_filters=[2, 3, 4, 5],
    fg_filters=[2, 3, 4, 5]
  )

  """No created sorted layers, ao montar os patches daquela camada, faço uma
   etapa adicional para reordanar. Preciso acrescentar uma nova etapa de reordenação."""

  # Moves layer folders to output folders
  layer_folders = [folder for folder in os.listdir(".") if "layer" in folder]
  for layer_folder in layer_folders:
    dst_layer_folder = Path(args.output_dir) / Path(layer_folder)
    # Verifies if layer folder exists, and deletes it if so
    if os.path.exists(dst_layer_folder):
      sh.rmtree(dst_layer_folder)
    sh.move(layer_folder, dst_layer_folder)