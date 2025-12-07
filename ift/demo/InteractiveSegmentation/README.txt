Interactive Segmentation:

*GUI Dependencies:
ift
tkinter

*Additional dependencies for experiments:
numpy
matplotlib

*Compiling:
Check the variables PYTHON_INCLUDE and PYTHON_LIB set in the Makefile, define the environment variable NEWIFT_DIR (path to ift folder), then "make".

*Running:
$python is.py

*File Conventions:
- Image folders contain the images (.ppm) in the database. For each annotated image i.ppm, a file i.markers is created to store the corresponding markers.
- Image folders contain a folder called gt/, where the ground truth images are stored (.pgm), with the same name as its corresponding image. 
- The ground truth image should be white in the object region and black in the background. 

-----

pyift:

The wrapper is created from the file pyift.i. Check its comments for more details on how to modify pyift. Check the Makefile for compilation and linking details.