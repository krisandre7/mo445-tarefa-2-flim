#!/usr/bin/python2.7
import os
import pyift
import pygc
import argparse
#import matplotlib
import numpy
#matplotlib.use('Agg')
#import pylab
from sklearn import metrics
import time
from skimage import io
import subprocess
import tempfile

class SegmentationMethod:
    """
    Base class for segmentation methods.
    Its subclasses should implement the method self.segment
    """
    def __init__(self, seg_params):
        """ 
        SegmentationMethod constructor.
        
        Keyword arguments:
            seg_params - dict containing the parameters needed by the segmentation
        method.
        """
        self.seg_params = seg_params
    
    def segment(self, image, seed):
        """ 
        Segments an image given a labeled set of seeds.
        This method should be implemented by subclasses of this class.
        
        Keyword arguments:
            image - iftImage to be segmented
            seed - iftLabeledSet containing the seeds
        
        Returns: iftImage representing the segmentation.
        """
        raise Exception('SegmentationMethod.segment must be implemented')
    
class SuperpixelSegmentation(SegmentationMethod):
    """  Superpixel segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class SuperpixelSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters:
                spatial_radius - controls adjacency radius (float)
                volume_threshold - controls superpixel volume (float)
                
                alpha - list containing 7 elements weighting: mean band1, mean band2, mean band3, relative area, color histogram intersection, lbp histogram, rectangularity
                colorspace - colorspace integer code. ycbcr: 0, rgb: 1, lab: 4
                bins_per_band - bins per band for color histogram
        """
        SegmentationMethod.__init__(self, seg_params)
        
        self.spatial_radius = seg_params['spatial_radius']
        self.volume_threshold = seg_params['volume_threshold']
        
        self.alpha = seg_params['alpha']
        self.colorspace = seg_params['colorspace']
        self.bins_per_band = seg_params['bins_per_band']
        
    def segment(self, image, seed):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        starting = time.time()   
        label_image = pyift.iftCreateRefinedLabelImage(image, seed, self.spatial_radius, self.volume_threshold, 2, 0.2)
        
        dataset = pyift.iftSupervoxelsToSelectiveSearchDataset(image, label_image, self.bins_per_band, self.colorspace)
        dataset.set_alpha(self.alpha)
                
        adj_relation = pyift.iftCircular(1)
        region_graph = pyift.iftRegionGraphFromLabelImage(label_image,dataset,adj_relation)
        
        ending = time.time()
        #print("superpixel setup time: {0}s".format(ending - starting))
        
        starting = time.time()        
        pyift.iftSuperpixelClassification(region_graph, label_image, seed);  
                
        classification_image = pyift.iftSuperpixelLabelImageFromDataset(label_image, dataset)
        
        ending = time.time()
        
        #print("superpixel segmentation time: {0}s".format(ending - starting))
        
        pyift.iftWriteSeedsOnImage(classification_image,seed)
        
        return classification_image
    
    def __repr__(self):
        return "super_seg"
    
class PixelSegmentation(SegmentationMethod):
    """  Pixel segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class PixelSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters:
                alpha - list containing 3 elements weighting: band 1, band 2, band 3
                colorspace - colorspace integer code. ycbcr: 0, rgb: 1, lab: 4
                spatial_radius: adjacency radius
        """
        SegmentationMethod.__init__(self, seg_params)
        
        self.spatial_radius = seg_params['spatial_radius']
        self.alpha = seg_params['alpha']
        self.colorspace = seg_params['colorspace']
    
    def segment(self, image, seed):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        starting = time.time()
        a = pyift.iftCircular(self.spatial_radius)
        
        conv_image = pyift.iftConvertColorSpace(image, 0, self.colorspace);
        
        dataset = pyift.iftImageToDataSet(conv_image)
        dataset.set_alpha(self.alpha)
        
        ending = time.time()
        #print("pixel setup time: {0}s".format(ending - starting))
        
        starting = time.time()
        label_image = pyift.iftWatershedOnPixelDist(dataset,a,seed)
        ending = time.time()
        #print("pixel segmentation time: {0}s".format(ending - starting))
        
        return label_image
    
    def __repr__(self):
        return "pixel_seg"
    
class PixelGradSegmentation(SegmentationMethod):
    """  Pixel (gradient) segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class PixelGradSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters:
                spatial_radius: adjacency radius used to create basins
        """
        SegmentationMethod.__init__(self, seg_params)
        
        self.spatial_radius = seg_params['spatial_radius']
    
    def segment(self, image, seed):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        starting = time.time()
        
        a = pyift.iftCircular(self.spatial_radius)
        adj_relation = pyift.iftCircular(self.spatial_radius);
        basins = pyift.iftImageBasins(image,adj_relation)
        
        ending = time.time()
        #print("pixel grad setup time: {0}s".format(ending - starting))
        
        starting = time.time()
        label_image = pyift.iftWatershed(basins,adj_relation,seed)
        ending = time.time()
        #print("pixel grad segmentation time: {0}s".format(ending - starting))
        
        return label_image
    
    def __repr__(self):
        return "pixel_grad_seg"
    
    
class GCPixelGradSegmentation(SegmentationMethod):
    """  Graph-cut pixel gradient segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class GCPixelGradSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters are:
                spatial_radius - spatial radius used to create basins
                beta - graph-cut parameter to compute arc-weights
        """
        SegmentationMethod.__init__(self, seg_params)
        self.spatial_radius = seg_params['spatial_radius']
        self.beta = seg_params['beta']
    
    def segment(self, image, seed):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        starting = time.time()
        
        adj_relation = pyift.iftCircular(self.spatial_radius);
        
        basins = pyift.iftImageBasins(image,adj_relation)
        label_image = pygc.graph_cut_pixel_grad_segmentation(basins,seed, self.beta)
        
        ending = time.time()
        
        #print(" gc pixel grad TOTAL time: {0}s".format(ending - starting))
        
        return label_image
    
    def __repr__(self):
        return "gc_pixel_grad_seg"
    
class GCPixelSegmentation(SegmentationMethod):
    """  Graph-cut pixel segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class GCPixelGradSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters are:
                spatial_radius - adjacency radius
                beta - graph-cut parameter to compute arc-weights
                
                alpha - list containing 3 elements weighting: band 1, band 2, band 3
                colorspace - colorspace integer code. ycbcr: 0, rgb: 1, lab: 4
        """
        SegmentationMethod.__init__(self, seg_params)
        self.beta = seg_params['beta']
        self.spatial_radius = seg_params['spatial_radius']
        self.colorspace = seg_params['colorspace']
        self.alpha = seg_params['alpha']
    
    def segment(self, image, seed):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        starting = time.time()
        a = pyift.iftCircular(self.spatial_radius)
        
        conv_image = pyift.iftConvertColorSpace(image, 0, self.colorspace);
        dataset = pyift.iftImageToDataSet(conv_image)
        dataset.set_alpha(self.alpha)
        
        label_image = pygc.graph_cut_pixel_segmentation(dataset, a, seed, self.beta)
        
        ending = time.time()
        
        #print(" gc pixel TOTAL time: {0}s".format(ending - starting))
        
        return label_image
    
    def __repr__(self):
        return "gc_pixel_seg"
    
class GCSuperSegmentation(SegmentationMethod):
    """  Graph-cut superpixel segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class GCPixelGradSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters are:
                spatial_radius: adjacency radius used to create the basins
                volume_threshold: threshold for volume closing
                beta - graph-cut parameter to compute arc-weights
                
                alpha - list containing 7 elements weighting: mean band1, mean band2, mean band3, relative area, color histogram intersection, lbp histogram, rectangularity
                colorspace - colorspace integer code. ycbcr: 0, rgb: 1, lab: 4
                bins_per_band - bins per band for color histogram
        """
        SegmentationMethod.__init__(self, seg_params)
        self.beta = seg_params['beta']
        self.volume_threshold = seg_params['volume_threshold']
        self.spatial_radius = seg_params['spatial_radius']
        
        self.alpha = seg_params['alpha']
        self.colorspace = seg_params['colorspace']
        self.bins_per_band = seg_params['bins_per_band']
    
    def segment(self, image, seed):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        starting = time.time()
        label_image = pyift.iftCreateRefinedLabelImage(image, seed, self.spatial_radius, self.volume_threshold, 2, 0.2)
        
        dataset = pyift.iftSupervoxelsToSelectiveSearchDataset(image, label_image, self.bins_per_band, self.colorspace)
        dataset.set_alpha(self.alpha)
                
        adj_relation = pyift.iftCircular(1)
        region_graph = pyift.iftRegionGraphFromLabelImage(label_image,dataset,adj_relation)
        
        segmentation =  pygc.graph_cut_superpixel_segmentation(region_graph, label_image, seed, self.beta)
        pyift.iftWriteSeedsOnImage(segmentation,seed)
        
        ending = time.time()
        
        #print(" gc super TOTAL time: {0}s".format(ending - starting))
        
        return segmentation
    
    def __repr__(self):
        return "gc_super_seg"

class SLICSuperpixelSegmentation(SegmentationMethod):
    """  Superpixel (SLIC) segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class SLICSuperpixelSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters:
                compactness : controls the compactness of the superpixels
                nregions: controls the number of superpixels 
                
                alpha - list containing 7 elements weighting: mean band1, mean band2, mean band3, relative area, color histogram intersection, lbp histogram, rectangularity
                colorspace - colorspace integer code. ycbcr: 0, rgb: 1, lab: 4
                bins_per_band - bins per band for color histogram
        """
        SegmentationMethod.__init__(self, seg_params)
        
        self.compactness = seg_params['compactness']
        self.nregions = seg_params['nregions']
        
        self.alpha = seg_params['alpha']
        self.colorspace = seg_params['colorspace']
        self.bins_per_band = seg_params['bins_per_band']
        
    def segment(self, image, seed):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        starting = time.time()   
        
        (tmp_input_fd, tmp_input_fn) = tempfile.mkstemp()
        (tmp_output_fd, tmp_output_fn) = tempfile.mkstemp()
        
        try:
            pyift.iftWriteImageP6(image, tmp_input_fn)
            subprocess.Popen(["./slic_app.py", tmp_input_fn, tmp_output_fn, str(self.nregions), str(self.compactness)]).wait()
        finally:
            label_image = pyift.iftReadImageP2(tmp_output_fn)
            os.close(tmp_input_fd)
            os.close(tmp_output_fd)
            os.remove(tmp_input_fn)
            os.remove(tmp_output_fn)
        
        dataset = pyift.iftSupervoxelsToSelectiveSearchDataset(image, label_image, self.bins_per_band, self.colorspace)
        dataset.set_alpha(self.alpha)
                
        adj_relation = pyift.iftCircular(1)
        region_graph = pyift.iftRegionGraphFromLabelImage(label_image,dataset,adj_relation)

        ending = time.time()

        #print("slic superpixel setup time: {0}s".format(ending - starting))
        
        starting = time.time()        
        pyift.iftSuperpixelClassification(region_graph, label_image, seed);  
                
        classification_image = pyift.iftSuperpixelLabelImageFromDataset(label_image, dataset)
        
        ending = time.time()
        
        print("slic superpixel segmentation time: {0}s".format(ending - starting))
        
        pyift.iftWriteSeedsOnImage(classification_image,seed)
        
        return classification_image
    
    def __repr__(self):
        return "slic_super_seg"
    
class SLICSuperpixelGeodesicSegmentation(SegmentationMethod):
    """  Superpixel (SLIC) segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class SLICSuperpixelSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters:
                compactness : controls the compactness of the superpixels
                nregions: controls the number of superpixels 
                
                alpha - list containing 7 elements weighting: mean band1, mean band2, mean band3, relative area, color histogram intersection, lbp histogram, rectangularity
                colorspace - colorspace integer code. ycbcr: 0, rgb: 1, lab: 4
                bins_per_band - bins per band for color histogram
                beta - beta for arc weight computation
        """
        SegmentationMethod.__init__(self, seg_params)
        
        self.compactness = seg_params['compactness']
        self.nregions = seg_params['nregions']
        
        self.alpha = seg_params['alpha']
        self.colorspace = seg_params['colorspace']
        self.bins_per_band = seg_params['bins_per_band']
        self.beta = seg_params['beta']
        
    def segment(self, image, seed):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        starting = time.time()   
        
        (tmp_input_fd, tmp_input_fn) = tempfile.mkstemp()
        (tmp_output_fd, tmp_output_fn) = tempfile.mkstemp()
        
        try:
            pyift.iftWriteImageP6(image, tmp_input_fn)
            subprocess.Popen(["./slic_app.py", tmp_input_fn, tmp_output_fn, str(self.nregions), str(self.compactness)]).wait()
        finally:
            label_image = pyift.iftReadImageP2(tmp_output_fn)
            os.close(tmp_input_fd)
            os.close(tmp_output_fd)
            os.remove(tmp_input_fn)
            os.remove(tmp_output_fn)
        
        dataset = pyift.iftSupervoxelsToSelectiveSearchDataset(image, label_image, self.bins_per_band, self.colorspace)
        dataset.set_alpha(self.alpha)
                
        adj_relation = pyift.iftCircular(1)
        region_graph = pyift.iftRegionGraphFromLabelImage(label_image,dataset,adj_relation)

        ending = time.time()

        #print("slic superpixel setup time: {0}s".format(ending - starting))
        
        starting = time.time()        
        pyift.iftSuperpixelClassificationGeodesic(region_graph, label_image, seed, self.beta);  
                
        classification_image = pyift.iftSuperpixelLabelImageFromDataset(label_image, dataset)
        
        ending = time.time()
        
        print("slic superpixel geodesic segmentation time: {0}s".format(ending - starting))
        
        pyift.iftWriteSeedsOnImage(classification_image,seed)
        
        return classification_image
    
    def __repr__(self):
        return "slic_super_geodesic_seg"


class SLICGCSuperSegmentation(SegmentationMethod):
    """  Graph-cut superpixel segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class GCPixelGradSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters are:
                compactness : controls the compactness of the superpixels
                nregions: controls the number of superpixels 
                
                beta - graph cut beta
                alpha - list containing 7 elements weighting: mean band1, mean band2, mean band3, relative area, color histogram intersection, lbp histogram, rectangularity
                colorspace - colorspace integer code. ycbcr: 0, rgb: 1, lab: 4
                bins_per_band - bins per band for color histogram
        """
        SegmentationMethod.__init__(self, seg_params)
        
        self.compactness = seg_params['compactness']
        self.nregions = seg_params['nregions']
        
        self.beta = seg_params['beta']
        
        self.alpha = seg_params['alpha']
        self.colorspace = seg_params['colorspace']
        self.bins_per_band = seg_params['bins_per_band']
    
    def segment(self, image, seed):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        starting = time.time()   
        
        (tmp_input_fd, tmp_input_fn) = tempfile.mkstemp()
        (tmp_output_fd, tmp_output_fn) = tempfile.mkstemp()
        
        try:
            pyift.iftWriteImageP6(image, tmp_input_fn)
            subprocess.Popen(["./slic_app.py", tmp_input_fn, tmp_output_fn, str(self.nregions), str(self.compactness)]).wait()
        finally:
            label_image = pyift.iftReadImageP2(tmp_output_fn)
            os.close(tmp_input_fd)
            os.close(tmp_output_fd)
            os.remove(tmp_input_fn)
            os.remove(tmp_output_fn)
        
        dataset = pyift.iftSupervoxelsToSelectiveSearchDataset(image, label_image, self.bins_per_band, self.colorspace)
        dataset.set_alpha(self.alpha)
                
        adj_relation = pyift.iftCircular(1)
        region_graph = pyift.iftRegionGraphFromLabelImage(label_image,dataset,adj_relation)

        ending = time.time()
        
        print("gc slic superpixel setup time: {0}s".format(ending - starting))
        
        segmentation =  pygc.graph_cut_superpixel_segmentation(region_graph, label_image, seed, self.beta)
        
        ending = time.time()
        print("slic gc super TOTAL time: {0}s".format(ending - starting))
        
        pyift.iftWriteSeedsOnImage(segmentation,seed)
        
        return segmentation
    
    def __repr__(self):
        return "slic_gc_super_seg"
    
class Experiment:
    """
    Base class for the Experiments. 
    
    If the list of seeds can be created from the image and its ground truth
    before the experiments begin, the method Experiments.create_seeds should be
    overridden. Otherwise, the method Experiments.run should be overriden.
    
    Details about this experimentation protocol can be found in the article 
    Interactive Segmentation by Image Foresting Transform presented at SIBGRAPI
    2013.  
    """
    def __init__(self, exp_params):
        """ 
        Constructor for the class Experiment
        
        Keyword arguments:
            exp_param - dict containing the experiment parameters. Expected 
            parameters are: 
                - dataset_path: path to the dataset
                - number_iterations: number of iterations of the experiment
                - seeds_per_iteration: number of seeds chosen at each iteration
                - max_marker: maximum radius of the marker
                - min_marker: minimum radius of the marker
                - safe_distance: distance considered safe to draw a marker. Should
                be inversely proportional to the quality of the ground truth.
        """
        self.exp_params = exp_params
        
        self.current_path = os.path.expanduser(exp_params['dataset_path'])
        self.current_images = [f for f in os.listdir(os.path.join(self.current_path,'orig')) if f.endswith(".ppm")]
    
    def create_seeds(self, image, gt_image):
        """
        Creates seeds given an image and its ground truth. Should be overriden
        by subclasses.
        
        Keyword arguments:
            image - target iftImage
            gt_image - the ground truth for the image
        """
        
        raise Exception('Experiment.create_seeds must be implemented')
    
    def run(self, seg_method):
        """ Runs the experiment.
            
            Keyword arguments:
                seg_method - segmentation method, subclass of the class Segmentation.
                
            Returns: a list containing a dict of statistics by iteration.
            Each dict holds the statistics for mean precision, recall, fscore
            and accuracy at a given iteration.
        
        """
        stats = [ { 'precision': 0.0, 'recall' : 0.0, 'accuracy': 0.0, 'fscore' : 0.0 } for _ in range(self.exp_params['number_iterations'])]
        
        print("Testing parameters :{0}\n".format(seg_method.seg_params))
        
        number_iterations = self.exp_params['number_iterations']
        for image_path in self.current_images:
            print("Image : {0}".format(image_path))
            image = pyift.iftReadImageP6(os.path.join(self.current_path,'orig',image_path))
            gt_image = pyift.iftReadImageP5(os.path.join(self.current_path,"labels",image_path[:-3] + "pgm"))
            
            segmentation = None
            seeds_bitmap = pyift.iftCreateBMap(image.n)
            available_seeds = self.create_seeds(image, gt_image)
            seed_image = pyift.iftCreateImage(image.xsize,image.ysize, image.zsize)
            pyift.iftSetImage(seed_image, -1)
            
            for i in range(number_iterations):
                pyift.iftMarkersFromMisclassifiedSeeds(seed_image, available_seeds, seeds_bitmap, self.exp_params['seeds_per_iteration'], 2, gt_image, segmentation,self.exp_params['safe_distance'],self.exp_params['max_marker'],self.exp_params['min_marker'])
                seeds = pyift.iftLabeledSetFromSeedImage(seed_image)
                
                segmentation = seg_method.segment(image, seeds)
                if 'output_path' in self.exp_params:
                    self.save_image(self.exp_params['output_path'], image_path, i + 1,seg_method, image, seeds, segmentation)
                
                errors = pyift.iftSegmentationErrors(gt_image,segmentation)
                
                stats[i]['precision'] += pyift.iftPrecisionGivenErrors(errors)
                stats[i]['recall'] += pyift.iftRecallGivenErrors(errors)
                stats[i]['fscore'] += pyift.iftFScoreGivenErrors(errors)
                stats[i]['accuracy'] += pyift.iftAccuracyGivenErrors(errors) 
        
        for i in range(number_iterations):
            for k,v in stats[i].items():
                stats[i][k] = v/len(self.current_images)
                
        print("Finished testing parameters : {0}".format(seg_method.seg_params))
                
        return stats
    
    def save_image(self, output_dir, image_path, iteration, seg_method, image, seeds, segmentation):
        markers_image = pyift.paint_seeds_on_image(image, seeds)
        segmentation_image = pyift.paint_segmentation_on_image(markers_image, segmentation)

        pyift.iftWriteImageP6(segmentation_image, os.path.join(output_dir, image_path[:-4]  + '_' + str(seg_method) + '_' + str(iteration) + '_segmentation.ppm'))
    
class PixelRobotExperiment(Experiment):
    """ Pixel robot experiment. Subclass of Experiment. """
    def __init__(self, exp_params, robot_params):
        """ Constructor for the class PixelRobotExperiment.
        
        Keyword arguments:
            exp_params: parameters for the experiment (see Experiment.__init__)
            robot_params: parameters for the robot. The expected parameter is:
                - border_distance: distance for erosion/dilation (float)
        """
        Experiment.__init__(self, exp_params)
        self.robot_params = robot_params
    
    def create_seeds(self, image, gt_image):
        """ Creates seeds given an image and its ground truth.
         (See Experiment.create_seeds) """
        adj_relation = pyift.iftCircular(1.5)
        basins = pyift.iftImageBasins(image,adj_relation)
        
        return pyift.iftBorderMarkersForPixelSegmentation(basins,gt_image,self.robot_params['border_distance'])
    
    def __repr__(self):
        return "pixel_robot"
            
class SuperpixelRobotExperiment(Experiment):
    """ Superpixel robot experiment. Subclass of Experiment. """
    def __init__(self, exp_params, robot_params):
        """ Constructor for the class SuperpixelRobotExperiment.
        
        Keyword arguments:
            exp_params: parameters for the experiment (see Experiment.__init__)
            robot_params: parameters for the robot. The expected parameter is:
                - volume_threshold: integer that controls the volume of the super
                pixels
        """
        Experiment.__init__(self, exp_params)
        self.robot_params = robot_params
    
    def create_seeds(self, image, gt_image):
        """ Creates seeds given an image and its ground truth.
         (See Experiment.create_seeds) """
        adj_relation = pyift.iftCircular(1.5)
        basins = pyift.iftImageBasins(image,adj_relation)
        marker = pyift.iftVolumeClose(basins,self.robot_params['volume_threshold'])
        label_image = pyift.iftWaterGray(basins,marker,adj_relation)
        dataset = pyift.iftSupervoxelsToDataSet(image, label_image)
        if dataset.nfeats == 3:
            dataset.set_alpha([0.2,1.0,1.0])
          
        return pyift.iftBorderMarkersForSuperpixelSegmentation(label_image,gt_image, dataset)

    def __repr__(self):
        return "superpixel_robot"
            
class GeodesicRobotExperiment(Experiment):
    """ Geodesic robot experiment. Subclass of Experiment. """
    def __init__(self, exp_params, robot_params):
        """ Constructor for the class SuperpixelRobotExperiment.
        
        Keyword arguments:
            exp_params: parameters for the experiment (see Experiment.__init__)
            robot_params: parameters for the robot. 
        """
        Experiment.__init__(self, exp_params)
        
    def create_seeds(self, image, gt_image, segmentation):
        """ Creates seeds given an image, its ground truth and the current
        segmentation
        
        Keyword arguments:
            image - target iftImage
            gt_image - the ground truth of the image (iftImage)
            segmentation - the segmentation of the image (iftImage)
            
        Returns: seeds (iftLabeledSet)
        """
        return pyift.iftGeodesicMarkersForSegmentation(gt_image, segmentation)
        
    def run(self, seg_method):
        """ 
        Runs the experiments given a segmentation method. 
        
        Keyword arguments:
            seg_method - segmentation method, subclass of the class Segmentation.
            
        Returns: a list containing a dict of statistics by iteration.
        Each dict holds the statistics for mean precision, recall, fscore
        and accuracy at a given iteration.
        """
        
        stats = [ { 'precision': 0.0, 'recall' : 0.0, 'accuracy': 0.0, 'fscore' : 0.0 } for _ in range(self.exp_params['number_iterations'])]
        
        print("Testing parameters :{0}".format(seg_method.seg_params))
        
        number_iterations = self.exp_params['number_iterations']
        for image_path in self.current_images:
            print("Image : {0}".format(image_path))
            image = pyift.iftReadImageP6(os.path.join(self.current_path,'orig',image_path))
            gt_image = pyift.iftReadImageP5(os.path.join(self.current_path,'labels',image_path[:-3] + "pgm"))
            
            segmentation = None
            seeds_bitmap = pyift.iftCreateBMap(image.n)
            seed_image = pyift.iftCreateImage(image.xsize,image.ysize, image.zsize)
            pyift.iftSetImage(seed_image, -1)
            
            for i in range(number_iterations):
                available_seeds = pyift.iftGeodesicMarkersForSegmentation(gt_image, segmentation)
                pyift.iftMarkersFromMisclassifiedSeeds(seed_image, available_seeds, seeds_bitmap, self.exp_params['seeds_per_iteration'], 2, gt_image, segmentation,self.exp_params['safe_distance'],self.exp_params['max_marker'],self.exp_params['min_marker'])
                seeds = pyift.iftLabeledSetFromSeedImage(seed_image)
                
                segmentation = seg_method.segment(image, seeds)
                if 'output_path' in self.exp_params:
                    self.save_image(self.exp_params['output_path'], image_path, i+1, seg_method, image, seeds, segmentation)
                
                errors = pyift.iftSegmentationErrors(gt_image,segmentation)
                
                stats[i]['precision'] += pyift.iftPrecisionGivenErrors(errors)
                stats[i]['recall'] += pyift.iftRecallGivenErrors(errors)
                stats[i]['fscore'] += pyift.iftFScoreGivenErrors(errors)
                stats[i]['accuracy'] += pyift.iftAccuracyGivenErrors(errors) 
        
        for i in range(number_iterations):
            for k,v in stats[i].items():
                stats[i][k] = v/len(self.current_images)
                
        print("Finished testing parameters : {0}".format(seg_method.seg_params))

        return stats
    
    def __repr__(self):
        return "geodesic_robot"
    
def main():
    """
    Main method. Runs segmentation experiments, based on robot users, on a 
    single database. Database conventions are described in the README.txt file
     included in this folder.
    
    The experiment setup is defined by the variable exp_params in this
    function.
    
    The robots are implemented as subclasses of the class Experiment.
    
    The segmentation methods are implemented as subclasses of the class
    Segmentation. 
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type = str, help = 'Path to the dataset folder')
    parser.add_argument('output_path', type = str, help = 'Path to the output folder')
    args = parser.parse_args()
    
    exp_params = {   
                  'dataset_path'         : os.path.expanduser(args.dataset_path),
                  'number_iterations' : 20,
                  'seeds_per_iteration'  : 2,
                  'max_marker'           : 8,
                  'min_marker'           : 1,
                  'safe_distance'        : 2
                  #'output_path'         : 'tmp'
                 }
    
    colorspace_code = { 'ycbcr' : 0, 'rgb' : 1, 'lab' : 4}    
    
    superpixel_robot = SuperpixelRobotExperiment(exp_params, { 'volume_threshold' : 15000 })
    pixel_robot = PixelRobotExperiment(exp_params, { 'border_distance' : 15 })
    geodesic_robot= GeodesicRobotExperiment(exp_params, {})
    
    superpixel_seg = SuperpixelSegmentation({'alpha': [2.178099212419126e-06, 6.6003006436943206e-06, 6.6003006436943206e-06, 0.0, 0.0, 0.0, 0.0], 'colorspace': 4, 'bins_per_band': 8, 'volume_threshold': 50, 'spatial_radius': 3.0})
    pixel_seg = PixelSegmentation({'alpha': [0.66, 1.0, 1.0], 'colorspace': 4, 'spatial_radius': 3.0})
    gc_super_seg = GCSuperSegmentation({'spatial_radius': 3.0, 'colorspace': 0, 'beta': 7, 'bins_per_band': 8, 'volume_threshold': 100, 'alpha': [1.1111111111111112e-05, 1.1111111111111112e-05, 1.1111111111111112e-05, 0.0, 1.0, 1.0, 0.0]})
    gc_seg = GCPixelSegmentation({'alpha': [1.0, 1.0, 1.0], 'beta': 9, 'colorspace': 1, 'spatial_radius': 3.0})
    
    slic_super_seg = SLICSuperpixelSegmentation({'compactness' : 10, 'nregions' : 600, 'alpha': [0.3, 1, 1, 0.0, 0.0, 0.0, 0.0], 'colorspace': 0, 'bins_per_band': 8})
    slic_gc_seg = SLICGCSuperSegmentation({'compactness' : 20, 'nregions' : 600, 'colorspace': 0, 'beta': 10, 'bins_per_band': 8, 'alpha': [1.1111111111111112e-05, 1.1111111111111112e-05, 1.1111111111111112e-05, 0.0, 1.0, 1.0, 0.0]})
    
    slic_super_geodesic_seg = SLICSuperpixelGeodesicSegmentation({'compactness' : 10, 'beta' : 20, 'nregions' : 600, 'alpha': [0.3, 1, 1, 0.0, 0.0, 0.0, 0.0], 'colorspace': 0, 'bins_per_band': 8})
    
    #pylab.figure(1,figsize = (18,18), dpi = 80)
    
    robots = [superpixel_robot, pixel_robot, geodesic_robot]
    segmentations = [superpixel_seg,gc_super_seg,pixel_seg,gc_seg]
    #FIXME
    segmentations = [slic_super_geodesic_seg]
    
    for i,robot in enumerate(robots):
        for j,segmentation in enumerate(segmentations):
            print("Running experiment.")
            stats = robot.run(segmentation)
            
            print("Results for {0}, {1}".format(robot,segmentation))
            for iter in range(exp_params['number_iterations']):
                print("Iteration {0}, f-score: {1:.3f}".format(iter+1, stats[iter]['fscore']))
            
            fscores = [ s['fscore'] for s in stats]
            auc = metrics.auc(numpy.linspace(0.,1.,len(fscores)),fscores)
            print('Area under curve: {0}'.format(auc))
            
            if args.output_path:
                numpy.save(os.path.join(args.output_path,'fscores_{0}_{1}'.format(robot,segmentation)),fscores)
                #pylab.subplot(len(robots),len(segmentations), i*len(segmentations) + j + 1)
                #pylab.title('{0}/{1} (auc: {2:.3f})'.format(robot,segmentation,auc))
                #pylab.xlabel('Iteration')
                #pylab.ylabel('Mean f-score')
                #pylab.ylim(0.5,1.0)
                #pylab.plot(range(1,len(fscores) + 1),fscores,'-o')
    
    if args.output_path:           
        None
        #pylab.tight_layout()  
        #pylab.savefig(os.path.join(args.output_path,'graph.pdf'))

if __name__ == '__main__':
    main()
