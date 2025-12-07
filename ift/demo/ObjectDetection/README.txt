Object Detection

The objective of this project is to detect objects of interest in images from the Pascal VOC dataset.

This module is organized in three submodules:

pascal.py: contains the code necessary to deal with the dataset (e.g. extract positive and negative samples for training)
param_eval.py: contains the code necessary to choose the parameters for the convolutional network that will be used to extract features from regions and train a region classifier.
detection.py: contains the code necessary to evaluate the convolutional network and its associated classifier.
recall_eval.py: contains the code necessary to evaluate candidate generation strategies
---

Dependencies:

swig
numpy
sklearn
skimage
lxml
joblib
pyift*

---

* pyift:

This module allows access to functions from libift. You need to compile this module by entering the directory pyift and typing "make". You will need a working copy of libift installed in $NEWIFT_DIR and the python headers. Make sure you are using the correct version of the headers.

---

Pascal classes:



