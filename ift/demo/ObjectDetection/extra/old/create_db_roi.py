#!/usr/bin/python2
import os
import numpy
from lxml import etree
from skimage import transform
from skimage import io

def main():
    """
    Given the dataset PASCAL 2012, creates a dataset following libift conventions.

    Firstly, you should download the database and the developer kit:
    http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2012/VOCtrainval_11-May-2012.tar
    http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2012/VOCdevkit_18-May-2011.tar

    After extracting the files, you should run this script from the directory 
    containing VOCdevkit/.

    This script will create 4 directories:
        - orig: contains png files with the regions of interest extracted from 
        all the images. Image names follow the convention [class_id]_[region_id].ppm
        - train_c: contains symlinks to the ppm files inside orig/ defined as 
        training images by the competition protocol
        - val_c: contains symlinks to the ppm files inside orig/ defined as
        validation images by the competition protocol
        - extra: contains the map from class ids to class names and the map from
        regions to images 

    Notes: 
    This script follows the advice given by the Pascal documentation and
    ignores the few images tagged as 'difficult'.

    Some objects are occluded and truncated. This is indicated in the annotation
    file. However, at the moment, we don't use such information.

    Dependencies: lxml (xml parsing), numpy (image manipulation), skimage (image
    manipulation).

    """
    output_format = 'jpg'

    #Creates directories
    for folder in ['orig','train_c','val_c','extra']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    #Reads image lists
    image_list = open('VOCdevkit/VOC2012/ImageSets/Main/trainval.txt').readlines()
    train_image_set = set(open('VOCdevkit/VOC2012/ImageSets/Main/train.txt').readlines())
    val_image_set = set(open('VOCdevkit/VOC2012/ImageSets/Main/val.txt').readlines())

    #Defines the paths to files
    image_base = 'VOCdevkit/VOC2012/JPEGImages'
    annot_base = 'VOCdevkit/VOC2012/Annotations'

    #Dictionaries to hold maps from class names to numbers and number of objects per class
    class_ids = {} 
    class_counts = {}

    #Creates a file to store a map from output images (regions) to original images
    region_to_image_map = open('extra/region_to_image.txt','w')

    for i,img_name in enumerate(image_list):
        print("Processing image {0}. ({1}/{2})".format(img_name.strip(),i+1,len(image_list)))

        #Opens the image
        image = io.imread(os.path.join(image_base,img_name.strip() + ".jpg"))
        #Opens the annotation file as a XML tree
        annot_tree = etree.parse(os.path.join(annot_base,img_name.strip() + ".xml"))
      
        #Obtains a list of XML elements containing every region of interest
        obj_trees = list(annot_tree.iter('object'))
        for obj in obj_trees:
            #Tests whether the region is tagged as difficult
            if int(obj.find('difficult').text):
                continue

            #Obtains the class for the region
            obj_class = obj.find('name').text
            if obj_class not in class_ids:
                class_ids[obj_class] = len(class_ids) + 1
                class_counts[obj_class] = 0
            class_counts[obj_class] += 1

            #Obtains the bounding box for the region
            bndbox_tree = obj.find('bndbox')
            xmax = int(bndbox_tree.find('xmax').text)
            ymax = int(bndbox_tree.find('ymax').text)
            xmin = int(bndbox_tree.find('xmin').text)
            ymin = int(bndbox_tree.find('ymin').text)

            #Extracts the region from the image and rescales it to its correct size
            obj = image[ymin:ymax+1,xmin:xmax+1]
            output_obj = obj

            #Defines a name to this region and saves it
            new_name = "{0:06d}_{1:07d}.{2}".format(class_ids[obj_class],class_counts[obj_class],output_format)
            io.imsave(os.path.join('orig',new_name),output_obj)

            #Writes to the region to image map.
            region_to_image_map.write("{0} {1}\n".format(new_name,img_name.strip()))

            #Creates a symlink in the appropriate directory
            if img_name in train_image_set:
                os.symlink(os.path.join('../orig',new_name), os.path.join('train_c',new_name))
            if img_name in val_image_set:
                os.symlink(os.path.join('../orig',new_name), os.path.join('val_c',new_name))

    region_to_image_map.close()
    
    #Sorts the class names by id and writes them to the class to id map
    fclass = open('extra/class_ids.txt','w')
    class_ids = dict((v,k) for (k,v) in class_ids.items())
    for k,v in class_ids.items():
        fclass.write('{0:02d} {1}\n'.format(k, v))
    fclass.close()

def clean():
    """ Deletes the directories created by the function main """
    import shutil
    for folder in ['extra','orig','train_c','val_c']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

if __name__ == "__main__":
    main()
