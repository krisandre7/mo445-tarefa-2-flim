#!/usr/bin/python2.7
import sys
import numpy
from skimage import segmentation
from skimage import io

def main():
    if len(sys.argv) != 5:
        print('usage: slic3d [INPUT] [OUTPUT] [NREGIONS] [COMPACTNESS]\n' + 
              '\tINPUT: input file \n' +
              '\tOUTPUT: output file \n' + 
              '\tNREGIONS: number of regions\n' + 
              '\tCOMPACTNESS : region compactness (default: 20)')
        sys.exit()
        
    input = sys.argv[1]
    output = sys.argv[2]
    
    nregions = int(sys.argv[3])
    compactness = float(sys.argv[4])
      
    image = numpy.load(input)
    zsize, ysize, xsize = image.shape
    io.imsave("middle_z.pgm", image[zsize/2])
                
    label = segmentation.slic(image, n_segments = nregions, compactness = compactness, multichannel = False) + 1
    print "Supervoxels: ", numpy.max(label)
    #for i in range(zsize):  # For all Z slices
    boundaries = segmentation.mark_boundaries(image[zsize/2], label[zsize/2], color=(1,1,0))
    io.imsave("boundaries.ppm", boundaries)
    
    label = numpy.array(label, dtype=numpy.int32)
    label.reshape(-1).tofile(output)

if __name__ == "__main__":
    main()
