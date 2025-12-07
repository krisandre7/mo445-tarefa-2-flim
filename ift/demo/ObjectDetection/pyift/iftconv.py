import pyift
import os

class IFTConvNetwork():
    """
    A class that encapsulates the libift implementation of a convolutional network.
    The most important method is transform, which receives an image path and returns a list of attributes.
    
    """
    def __init__(self, param = None, path = None):
        """
        Keyword arguments:
            param: a dictionary containing parameters that define a convolutional network
            path: path to convolutional network file
        """
        self.p = param
        if param:
            tmp_path = 'tmp_params.convnet'
            f = open(tmp_path,'w')
            f.write('CONVNET\n')
            f.write('NLAYERS {0}\n'.format(param["n_layers"]))
            f.write('INPUT_NORM_ADJ_PARAM {0}\n'.format(param["input_norm_size"]))
            
            f.write('INPUT_XSIZE {0}\n'.format(param["input_dim"][0]))
            f.write('INPUT_YSIZE {0}\n'.format(param["input_dim"][1]))
            f.write('INPUT_ZSIZE {0}\n'.format(param["input_dim"][2]))
            
            f.write('INPUT_NBANDS {0}\n'.format(param["n_bands_input"]))
                       
            f.write('K_BANK_ADJ_PARAM {0} {1} {2}\n'.format(*param["size_kernels"]))
            f.write('NKERNELS {0} {1} {2}\n'.format(*param["n_kernels"]))
            f.write('POOLING_ADJ_PARAM {0} {1} {2}\n'.format(*param["size_pooling"]))
            f.write('STRIDE {0} {1} {2}\n'.format(*param["stride"]))
            f.write('ALPHA {0} {1} {2}\n'.format(*param["alpha"]))
            f.write('NORM_ADJ_PARAM {0} {1} {2}\n'.format(*param["size_norm"]))
                       
            #Fixed parameters
            f.write('RESCALE 0\n')
            f.write('WITH_WEIGHTS 0\n')
            f.write('NEG_ACTIV_THRES  -1000000\n')
            f.write('POS_ACTIV_THRES   0\n')
            
            f.close()
            
            #This object is an iftConvNetwork
            self.ift_conv = pyift.iftReadConvNetwork(tmp_path)
            os.remove(tmp_path)
        elif path:
            self.ift_conv = pyift.iftReadConvNetwork(path)
        else:
            raise Exception('You must supply either param or path')
        
    def transform(self, image_path):
        """
        Extracts features from a given image.
        
        Keyword arguments:
            image: an iftImage from which to extract features.
        
        Returns: the extracted attributes as a list of floating point numbers
        """
        return pyift.iftPyExtractDeepFeatures(image_path, self.ift_conv)
    
    def save(self, path):
        pyift.iftSaveConvNetwork(self.ift_conv, path)
    
    def __repr__(self):
        if self.p:
            return self.p.__repr__()
        else:
            return "IFT convolutional network"
    
    def __exit__(self):
        """ Frees memory """
        pyift.iftPyDestroyConvNetwork(self.ift_conv)
