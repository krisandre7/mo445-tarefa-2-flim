#!/usr/bin/python2.7
from pyslic import slic
import sys
import numpy
from skimage import io
import pyift

def main():
    if len(sys.argv) != 5:
        print('usage: slic_app [INPUT] [OUTPUT] [NREGIONS] [COMPACTNESS]\n' + 
              '\tINPUT: input ppm file \n' +
              '\tOUTPUT: output pgm file \n' + 
              '\tNREGIONS: number of regions\n' + 
              '\tCOMPACTNESS : region compactness (default: 20)')
        sys.exit()
        
    input = sys.argv[1]
    output = sys.argv[2]
    nregions = int(sys.argv[3])
    compactness = int(sys.argv[4])
    
    im = numpy.array(io.imread(input))
    region_labels = slic.slic_n(im, nregions, compactness)
    
    label_image = pyift.iftCreateImage(region_labels.shape[1], region_labels.shape[0], 1)
    pyift.setIftImage(label_image, list(region_labels.reshape(-1) + 1))
    pyift.iftWriteImageP2(label_image, output)
    
    #FIXME
    adj_relation = pyift.iftCircular(1.5)
    adj_relation_2 = pyift.iftCircular(0.0)
    
    color = pyift.iftColor()
    color.set_values((255,0,0))
    
    ift_image_copy = pyift.iftReadImageP6(input)
    pyift.iftDrawBorders(ift_image_copy, label_image, adj_relation, color, adj_relation_2)
    pyift.iftWriteImageP6(ift_image_copy, output)

if __name__ == "__main__":
    main()
