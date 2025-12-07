This baseline is an adpation of the main idea of using the residual between the input image and its reconstruction image (using AAE, VAE, etc), followed by thresholding, for anomaly detection.
The papers do not present much information about their implementations.
Therefore, I coded a version of such works using a 2D Convolutional AutoEncoder with a given threshold.

Papers:
- Deep Generative Models in the Real-World: An Open Challenge from Medical Imaging - Chen, Xiaoran - ArXiv
- Deep Autoencoding Models for Unsupervised Anomaly Segmentation in Brain MR Images - Baur, Christoph - ArXiv
- Unsupervised Lesion Detection in Brain CT using Bayesian Convolutional Autoencoders - Pawlowski, Nick - MIDL 2018
- Unsupervised Detection of Lesions in Brain MRI using constrained adversarial auto-encoders - Chen, Xiaoran - MIDL 2018
