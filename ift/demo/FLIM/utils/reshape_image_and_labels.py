from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import os
import argparse
from pathlib import Path

"""
Execution sample:
python reshape_image_and_labels.py ~/msc/2_datasets/parasites/images \ 
~/msc/2_datasets/parasites/truelabels images_reshaped labels_reshaped 240,240
"""

def get_args():
    def image_size(arg):
        return tuple(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(
        prog="Reshapes input images and labels",
        description="Given an input image and label folder, reshape it to the "
                "desired size"
    )
    parser.add_argument("input_images_folder", help="Folder with input images")
    parser.add_argument("input_labels_folder", help="Folder with input labels")
    parser.add_argument("output_images_folder", help="Folder to save output images")
    parser.add_argument("output_labels_folder", help="Folder to save output labels")
    parser.add_argument("size", help="Reshaped image size", type=image_size)
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = get_args()
    images = os.listdir(args.input_images_folder)
    os.makedirs(args.output_images_folder, exist_ok=True)
    os.makedirs(args.output_labels_folder, exist_ok=True)
    for image in images:
        src_image_path = Path(args.input_images_folder) / Path(image)
        src_label_path = Path(args.input_labels_folder) / Path(image)
        dst_image_path = Path(args.output_images_folder) / Path(image)
        dst_label_path = Path(args.output_labels_folder) / Path(image)
        
        image = Image.open(src_image_path)
        label = Image.open(src_label_path)
    
        interpolation_image = transforms.Resize(
            size=args.size,
            interpolation=InterpolationMode.BILINEAR
        )
        interpolation_label = transforms.Resize(
            size=args.size,
            interpolation=InterpolationMode.NEAREST
        )
        
        image = interpolation_image(image)
        label = interpolation_label(label)
        
        image.save(dst_image_path)
        label.save(dst_label_path)