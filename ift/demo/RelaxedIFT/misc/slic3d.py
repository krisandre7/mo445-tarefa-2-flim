#!/usr/bin/python2.7
"""
Dependencies:

python2.7

skimage 0.10. 
    This version is not available on pip yet. Install following these instructions:
    Set your environment variable $PYTHONPATH to $HOME/lib/python2.7/site-packages. Don't forget to export it.
    Close all open terminals.
    Download https://github.com/scikit-image/scikit-image/zipball/master and extract to a folder.
    Run python setup.py install --prefix=$HOME on the folder created in the previous step.
    Open the python console, type:
        from skimage.version import version
        version
    The expected output is a version >= 0.10.
"""

import sys
import numpy
from skimage import segmentation
from skimage import io

def main():
    if len(sys.argv) != 8:
        print('usage: slic3d [XS] [YS] [ZS] [INPUT] [OUTPUT] [NREGIONS] [COMPACTNESS]\n' + 
              '\tXS: number of rows \n' +
              '\tYS: number of columns \n' +
              '\tZS: number of slices \n' +
              '\tINPUT: input file \n' +
              '\tOUTPUT: output file \n' + 
              '\tNREGIONS: number of regions\n' + 
              '\tCOMPACTNESS : region compactness (default: 20)')
        sys.exit()
        
    xs = int(sys.argv[1])
    ys = int(sys.argv[2])
    zs = int(sys.argv[3])
    
    input = sys.argv[4]
    output = sys.argv[5]
    
    nregions = int(sys.argv[6])
    compactness = float(sys.argv[7])
      
    array = numpy.fromfile(input, dtype=numpy.int32)
    array = numpy.array(array, dtype = float)/4096.
    image = array.reshape(zs,ys,xs)

    label = segmentation.slic(image, n_segments = nregions, compactness = compactness, multichannel = False) + 1
    
    label = numpy.array(label, dtype=numpy.int32)
    label.reshape(-1).tofile(output)

if __name__ == "__main__":
    main()