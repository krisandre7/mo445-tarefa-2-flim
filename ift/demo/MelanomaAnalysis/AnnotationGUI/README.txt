Usage:

This folder contains two applications designed to aid in skin lesion image annotation.

The application is.py can be used for interactive segmentation of skin lesion images. The user can load an image database (e.g. MelanomaSet2/orig) by clicking on "File -> Open Image Database" and selecting the folder that contains the ppm lesion images. The user can draw markers using the right and left mouse buttons. The segmentation is displayed after clicking on "Segment". The user can save the current segmentation, add new markers or remove all markers.  After saving the results, the user can move on to the next image by clicking on "Next". After segmenting all images in the database, the user can export the results to a zip file (File -> Export results). The segmentations are saved to a folder named labels, created in the parent of the folder containing the images. 

The application ia.py can be used for interactive annotation of skin lesion images that were previously segmented. The user can load an image database by clicking on "File -> Open Image Database", as in the previous example. A folder named "labels" must exist in the parent folder of the database. For instance, if the image database can be found in "MelanomaSet1/orig", the folder "MelanomaSet1" must contain the directory "labels".
The user can control the number of superpixels inside the lesion and annotate it using different labels. The superpixel size can be controlled through the parameter "Volume".  The user can add new labels and rename labels as desired. The user must keep in mind that the label's name should be consistent with all annotations in a given database. A confidence level (in %) can be associated to each marker drawn on the image. The spatial extent of a labeled superpixel can be seen by clicking on "OK" in the "Oversegmentation" frame.
The user can move on to the next image by clicking on "Next". The annotations are automatically saved when the user views another image or closes the application. After annotating all images in the database, the user can export the results to a zip image (File -> Export annotations).

---

Dependencies:

libift
python2.7
swig
Tkinter
skimage 0.10:
	 Most distributions are not up to date on skimage, so it must be installed using pip or following these instructions:
	Set your environment variable $PYTHONPATH to $HOME/lib/python2.7/site-packages. Don't forget to export it.
	Close all open terminals.
	Download https://github.com/scikit-image/scikit-image/zipball/master and extract to a folder.
	Run python setup.py install --prefix=$HOME on the folder created in the previous step.
	Open the python console, type:
		from skimage.version import version
		version
	The expected output is a version >= to 0.10.

---

Building:

Run "make" on the folder containing this file. This will compile the appropriate python module that links with libift.

---

Conventions:

All information is stored into folders inside the folder that contains the ppm lesion images. For instance, if the images are found under "MelanomaSet1/orig", four folders may be created by the two applications:

MelanomaSet1/orig/labels: This folder contains the segmentations (is.py)
MelanomaSet1/orig/overseg: This folder contains the oversegmentations (ia.py)
MelanomaSet1/orig/markers: This folder contains the markers drawn by the user (ia.py)
MelanomaSet1/orig/confidence: This folder contains the confidences by marker (ia.py)
MelanomaSet1/orig/params: This folder contains the oversegmentation parameters (ia.py) 

