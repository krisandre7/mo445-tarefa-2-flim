Usage:

The application create_dataset creates superpixel datasets from a folder containing a dataset annotated using AnnotationGui/ia.py.
One dataset is created for each image and stored under datasets/IMG_NAME.opf

The application eval_classification.py receives as input the path to the dataset containing the opf datasets created by create_dataset. It merges all feature vectors into a data matrix together with a vector representing the classes associated by the user.

---

Dependencies:

libift
python2
numpy
cython

---

Building:

Enter the folder cyift and run "make". Run "make create_dataset" to create .

---

Conventions:

All information is stored into folders inside the folder that contains the ppm lesion images. For instance, if the images are found under "MelanomaSet1/orig", four folders may be created by the two applications:

MelanomaSet1/labels: This folder contains the segmentations (is.py)
MelanomaSet1/orig/overseg: This folder contains the oversegmentations (ia.py)
MelanomaSet1/orig/markers: This folder contains the markers drawn by the user (ia.py)
MelanomaSet1/orig/confidence: This folder contains the confidences by marker (ia.py)
MelanomaSet1/orig/params: This folder contains the oversegmentation parameters (ia.py) 
