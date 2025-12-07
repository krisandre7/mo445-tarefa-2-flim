#!/usr/bin/python2.7
import os
import sys
import time
from pyift import *

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
    
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
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
        """
        SegmentationMethod.__init__(self, seg_params)
        
        self.spatial_radius = seg_params['spatial_radius']
        self.volume_threshold = seg_params['volume_threshold']
        
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
	
       	adj_relation = iftCircular(self.spatial_radius);
        basins = iftImageBasins(image,adj_relation)
	#basins = border_image
        marker = iftVolumeClose(basins,self.volume_threshold)
        
	#label_image = iftMergeWatergray(image,self.volume_threshold,800,outputwg);
        label_image = iftWaterGray(basins,marker,adj_relation)
	#steps = 3
	#vol_ratio = 0.5
	#label_image = iftCreateRefinedLabelImage(image, seed, self.spatial_radius,self.volume_threshold, steps, vol_ratio)
        
        dataset = iftSupervoxelsToDataSet(image, label_image)
        if dataset.nfeats == 3:
            dataset.set_alpha([0.2,1.0,1.0])
                
        adj_relation = iftCircular(1)
        region_graph = iftRegionGraphFromLabelImage(label_image,dataset,adj_relation)
        
        iftSuperpixelClassification(region_graph,label_image, seed);  
                
        classification_image = iftCreateImage(image.xsize,image.ysize,image.zsize)
        for p in range(0,classification_image.n):
            pixel_label = label_image[p][0] - 1
            color = dataset.get_sample_label(pixel_label)
            classification_image[p] = (color, 0, 0)
        
        iftWriteSeedsOnImage(classification_image,seed)
        
        return classification_image
    
    def __repr__(self):
        return "superpixel segmentation"

class SuperpixelSegmentationSyn(SegmentationMethod):
    """  Superpixel segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class SuperpixelSegmentationSyn.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters:
                spatial_radius - controls adjacency radius (float)
                volume_threshold - controls superpixel volume (float)
        """
        SegmentationMethod.__init__(self, seg_params)
        
        self.spatial_radius = seg_params['spatial_radius']
        self.volume_threshold = seg_params['volume_threshold']
	self.convnet_file = seg_params['convnet_file']
        
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        return iftSuperpixelSegmentationSyn(image,seed,self.spatial_radius,self.volume_threshold,self.convnet_file, num_iter, dataset_path, outputwg, border_image, outputbas)
    
    def __repr__(self):
        return "superpixel segmentation syn"
    
class SuperDeepSegmentation(SegmentationMethod):
    """  Deep superpixel segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class SuperDeepSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameters:
                spatial_radius - controls adjacency radius (float)
                volume_threshold - controls superpixel volume (float)
        """
        SegmentationMethod.__init__(self, seg_params)
        
        self.spatial_radius = seg_params['spatial_radius']
        self.volume_threshold = seg_params['volume_threshold']
        self.convnet_file = seg_params['convnet_file']
	self.convnet_file_basins = seg_params['convnet_file_basins']
	self.type_weights = seg_params['type_weights']
        
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        return iftSuperDeepSegmentation(image,seed,self.spatial_radius,self.volume_threshold,self.convnet_file,self.convnet_file_basins, num_iter, self.type_weights, dataset_path, outputwg, border_image, outputbas)    
            
    def __repr__(self):
        return "superpixel deep segmentation"
    
class PixelSegmentation(SegmentationMethod):
    """  Pixel segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class PixelSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. There are no expected parameters.
        """
        SegmentationMethod.__init__(self, seg_params)
    
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        a = iftCircular(1.5)
        
        dataset = iftImageToDataSet(image)
        if dataset.nfeats == 3:
            dataset.set_alpha([0.2,1.0,1.0])
        label_image = iftWatershedOnPixelDist(dataset,a,seed)
        
        return label_image
    
    def __repr__(self):
        return "pixel segmentation"
    
class PixelGradSegmentation(SegmentationMethod):
    """  Pixel (gradient) segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class PixelGradSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. There are no expected parameters.
        """
        SegmentationMethod.__init__(self, seg_params)
    
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        a = iftCircular(1.5)
        
        basins = iftEnhanceEdges(image,a,seed,0.0)
        label_image = iftWatershed(basins,a,seed)
        
        return label_image
    
    def __repr__(self):
        return "pixel gradient segmentation"

class PixelGradSegmentationSobel(SegmentationMethod):
    """  Pixel (gradient) segmentation sobel. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class PixelGradSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. There are no expected parameters.
        """
        SegmentationMethod.__init__(self, seg_params)
    
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        return iftPixelGradSegmentationSobel(image, seed, num_iter, dataset_path, outputwg, border_image, outputbas)
    
    def __repr__(self):
        return "pixel gradient segmentation sobel"

class PixelGradSegmentationEnhancement(SegmentationMethod):
    """  Pixel (gradient) segmentation enhancement. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class PixelGradSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. There are no expected parameters.
        """
        SegmentationMethod.__init__(self, seg_params)
	self.spatial_radius = seg_params['spatial_radius']
        self.volume_threshold = seg_params['volume_threshold']
    
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
     	
        return iftPixelGradSegmentationEnhancement(image,seed,self.spatial_radius,self.volume_threshold, num_iter, dataset_path, outputwg, border_image, outputbas)
    
    def __repr__(self):
        return "pixel gradient segmentation enhancement"

class PixelGradSegmentationEnhancementConvNet(SegmentationMethod):
    """  Pixel (gradient) segmentation enhancement. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class PixelGradSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. There are no expected parameters.
        """
        SegmentationMethod.__init__(self, seg_params)
	self.spatial_radius = seg_params['spatial_radius']
        self.volume_threshold = seg_params['volume_threshold']
        self.convnet_file = seg_params['convnet_file']
	self.convnet_file_basins = seg_params['convnet_file_basins']
    
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        """
        
        return iftPixelGradSegmentationEnhancementConvNet(image,seed,self.spatial_radius,self.volume_threshold, self.convnet_file, self.convnet_file_basins, num_iter, dataset_path, outputwg, border_image, outputbas)
    
    def __repr__(self):
        return "pixel gradient segmentation enhancemnent convnet"
    
class GCSegmentation(SegmentationMethod):
    """  Graph-cut segmentation. Subclass of SegmentationMethod """
    def __init__(self, seg_params):
        """ Constructor for the class GCSegmentation.
        
        Keyword arguments:
            seg_params - segmentation parameters. Expected parameter is:
                beta - graph-cut parameter to compute arc-weights
        """
        SegmentationMethod.__init__(self, seg_params)
    
    def segment(self, image, seed, num_iter, dataset_path, outputwg, border_image, outputbas):
        """ Segments an image given a labeled set of seeds. 
        (See SegmentationMethod.segment)
        
        Not implemented yet.
        """
        #TODO: GC Code
        a = iftCircular(1.5)
        
        basins = iftEnhanceEdges(image,a,seed,0.0)
        label_image = iftWatershed(basins,a,seed)
        
        return label_image
    
    def __repr__(self):
        return "graph cut segmentation"
    
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
        self.current_images = [f for f in os.listdir(self.current_path) if f.endswith(".ppm")]
    
    def saveImage(self, robot, image, imagen_name, i, seed_pix_image, pixel_cl_image):
	raise Exception('Experiment.saveImage must be implemented')
    
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
        dataset_path = "" #added
	outputwg = "" #added
	border_image = ""
        number_iterations = self.exp_params['number_iterations']
        for image_path in self.current_images:
            print("Image: {0}".format(image_path))
	    print("SUP1")

            image = iftReadImageP6(os.path.join(self.current_path,image_path))
            gt_image = iftReadImageP5(os.path.join(self.current_path,"gt/" + image_path[:-3] + "pgm"))
            border_image = iftReadImageP5(os.path.join(self.current_path,"border/" + image_path[:-3] + "pgm"))  #added

            segmentation = None
            seeds_bitmap = iftCreateBMap(image.n)
            available_seeds = self.create_seeds(image, gt_image)
            seed_image = iftCreateImage(image.xsize,image.ysize, image.zsize)
            
            for i in range(number_iterations):
                iftMarkersFromMisclassifiedSeeds(seed_image,available_seeds,seeds_bitmap,self.exp_params['seeds_per_iteration'],2,gt_image,segmentation,self.exp_params['safe_distance'],self.exp_params['max_marker'],self.exp_params['min_marker']) 
                seeds = iftLabeledSetFromSeedImage(seed_image)
                
                segmentation = seg_method.segment(image, seeds, i, dataset_path, outputwg, border_image)   #modified
                errors = iftSegmentationErrors(gt_image,segmentation)
		
		self.saveImage(format(self)[:3],image,image_path,i,seed_image,segmentation); #added
		
                stats[i]['precision'] += iftPrecisionGivenErrors(errors)
                stats[i]['recall'] += iftRecallGivenErrors(errors)
                stats[i]['fscore'] += iftFScoreGivenErrors(errors)
                stats[i]['accuracy'] += iftAccuracyGivenErrors(errors) 
        
        for i in range(number_iterations):
            for k,v in stats[i].items():
                stats[i][k] = v/len(self.current_images)
                
        return stats
    
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
    
    def saveImage(self, robot, image, imagen_name, i, seed_pix_image,
                  pixel_cl_image):
	output_path = "outputpix/"
        #Write partial markers to disk (debugging)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        colors = [(225, 1, 148), (76, 85, 255)]

        #Markers image
        image_pix_markers = iftCopyImage(image)
        for p in range(image.n):
            if seed_pix_image[p] != (0, 0, 0):
                image_pix_markers[p] = colors[seed_pix_image[p][0] - 1]

        #Segmentation image
        image_pix_seg = iftCopyImage(image_pix_markers)
        for p in range(image.n):
            if pixel_cl_image[p] == (1, 0, 0):
                image_pix_seg[p] = \
                    (image_pix_seg[p][0], colors[0][1], colors[0][2])
            else:
                image_pix_seg[p] = \
                    (image_pix_seg[p][0], colors[1][1], colors[1][2])
	
        outputname = \
            output_path + imagen_name[:-4] + "_" + \
            str(i+1) + "_" + robot + ".ppm"

        #iftWriteImageP6(image_pix_markers, outputname)
        iftWriteImageP6(image_pix_seg, outputname)    


    def create_seeds(self, image, gt_image):
        """ Creates seeds given an image and its ground truth.
         (See Experiment.create_seeds) """
        adj_relation = iftCircular(1.5)
        basins = iftImageBasins(image,adj_relation)
        
        return iftBorderMarkersForPixelSegmentation(basins,gt_image,self.robot_params['border_distance'])
    
    def __repr__(self):
        return "pixel robot"
            
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
    
    def saveImage(self, robot, image, imagen_name, i, seed_pix_image,
                  pixel_cl_image):
	output_path = "outputsup/"
        #Write partial markers to disk (debugging)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        colors = [(225, 1, 148), (76, 85, 255)]

        #Markers image
        image_pix_markers = iftCopyImage(image)
        for p in range(image.n):
            if seed_pix_image[p] != (0, 0, 0):
                image_pix_markers[p] = colors[seed_pix_image[p][0] - 1]

        #Segmentation image
        image_pix_seg = iftCopyImage(image_pix_markers)
        for p in range(image.n):
            if pixel_cl_image[p] == (1, 0, 0):
                image_pix_seg[p] = \
                    (image_pix_seg[p][0], colors[0][1], colors[0][2])
            else:
                image_pix_seg[p] = \
                    (image_pix_seg[p][0], colors[1][1], colors[1][2])
	
        outputname = \
            output_path + imagen_name[:-4] + "_" + \
            str(i+1) + "_" + robot + ".ppm"

        #iftWriteImageP6(image_pix_markers, outputname)
        iftWriteImageP6(image_pix_seg, outputname)

    def create_seeds(self, image, gt_image):
        """ Creates seeds given an image and its ground truth.
         (See Experiment.create_seeds) """
        adj_relation = iftCircular(1.5)
        basins = iftImageBasins(image,adj_relation)
        marker = iftVolumeClose(basins,self.robot_params['volume_threshold'])
        label_image = iftWaterGray(basins,marker,adj_relation)
        dataset = iftSupervoxelsToDataSet(image, label_image)
        if dataset.nfeats == 3:
            dataset.set_alpha([0.2,1.0,1.0])
        
        return iftBorderMarkersForSuperpixelSegmentation(label_image,gt_image, dataset)

    def __repr__(self):
        return "superpixel robot"
            
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
        return iftGeodesicMarkersForSegmentation(gt_image, segmentation)

    def saveImage(self, robot, image, imagen_name, i, seed_pix_image,
                  pixel_cl_image):
	output_path = "outputgeo/"
        #Write partial markers to disk (debugging)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        colors = [(225, 1, 148), (76, 85, 255)]

        #Markers image
        image_pix_markers = iftCopyImage(image)
        for p in range(image.n):
            if seed_pix_image[p] != (-1, 0, 0):
                image_pix_markers[p] = colors[seed_pix_image[p][0]]

        #Segmentation image
        image_pix_seg = iftCopyImage(image_pix_markers)
        for p in range(image.n):
            if pixel_cl_image[p] == (0, 0, 0):
                image_pix_seg[p] = \
                    (image_pix_seg[p][0], colors[0][1], colors[0][2])
            else:
                image_pix_seg[p] = \
                    (image_pix_seg[p][0], colors[1][1], colors[1][2])
	
        outputname = \
            output_path + imagen_name[:-4] + "_" + \
            str(i+1) + "_" + robot + ".ppm"

        #iftWriteImageP6(image_pix_markers, outputname)
        iftWriteImageP6(image_pix_seg, outputname)
        
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
        
        number_iterations = self.exp_params['number_iterations']
 	dataset_path = self.exp_params['dataset_path'] 	# added
	
	a = iftRectangular(7,7)		#added
	#mmkernel = iftUnsupLearnKernelsFromImages(dataset_path, a, 10, 0.01,1,0)
	#mmkernel = dataset_path
	
	all_iterations = [] 	#added
	num_img = 0		#added
        for image_path in self.current_images:
	    all_iterations.append([])	#added
            print("Image: {0}".format(image_path))
            #outputwg = "outputwg/" + image_path #added
	    #outputbas = "outputbas/" + image_path[:-3] + "pgm" #added
            image = iftReadImageP6(os.path.join(self.current_path,image_path))
            gt_image = iftReadImageP5(os.path.join(self.current_path,"gt/" + image_path[:-3] + "pgm"))
            border_image = iftReadImageP5(os.path.join(self.current_path,"gt/" + image_path[:-3] + "pgm"))  #added
	    crop = iftReadImageP6(os.path.join(self.current_path,"crop/" + image_path[:-3] + "ppm")) #added
            segmentation = None
            seeds_bitmap = iftCreateBMap(image.n)
            seed_image = iftCreateImage(image.xsize, image.ysize, image.zsize)
	    for p in range(image.n):
		seed_image[p] = (-1,0,0)
            for i in range(number_iterations):
		outputwg = "outputwg/" + image_path[:-4] + ".pgm" #added
		outputbas = "outputbas/" + image_path[:-4] + "_"+ str(i+1) +".pgm" #added
                available_seeds = iftGeodesicMarkersForSegmentation(gt_image, segmentation)
                iftMarkersFromMisclassifiedSeeds(seed_image,available_seeds,seeds_bitmap,self.exp_params['seeds_per_iteration'],2,gt_image,segmentation,self.exp_params['safe_distance'],self.exp_params['max_marker'],self.exp_params['min_marker']) 
		
                seeds = iftLabeledSetFromSeedImage(seed_image)
                segmentation = seg_method.segment(image, seeds, i, dataset_path, outputwg, border_image, outputbas) 	# modified
                errors = iftSegmentationErrors(gt_image,segmentation)
		
		#if (i == (number_iterations-1)):		#added
		self.saveImage(format(self)[:3],image,image_path,i,seed_image,segmentation); #added
		
                
                stats[i]['precision'] += iftPrecisionGivenErrors(errors)
                stats[i]['recall'] += iftRecallGivenErrors(errors)
                stats[i]['fscore'] += iftFScoreGivenErrors(errors)
                stats[i]['accuracy'] += iftAccuracyGivenErrors(errors) 
	        all_iterations[num_img].append(iftFScoreGivenErrors(errors))	#added
	    all_iterations[num_img].append(image_path)	#added
	    num_img = num_img +1 	#added
        print all_iterations  	#added
        for i in range(number_iterations):
            for k,v in stats[i].items():
                stats[i][k] = v/len(self.current_images)

        return stats
    
    def __repr__(self):
        return "geodesic robot"
    
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
    if len(sys.argv) != 4:
		print "Error. Please provide:"
		print "<convnet_file> <path_directory_images> <type_weights>"
		return 1
    # read input
    convnet_file = sys.argv[1]
    convnet_file_basins = ""
    database = sys.argv[2]
    type_weights = int(sys.argv[3])
    exp_params = {   
                  'dataset_path'         : database,
                  'number_iterations'    : 8,
                  'seeds_per_iteration'  : 8,
                  'max_marker'           : 5,
                  'min_marker'           : 1,
                  'safe_distance'        : 2
                 }
    start_time = time.time()
    print database
    print convnet_file
    print type_weights
    superpixel_robot = SuperpixelRobotExperiment(exp_params, { 'volume_threshold' : 15000 })
    pixel_robot = PixelRobotExperiment(exp_params, { 'border_distance' : 15 })
    geodesic_robot= GeodesicRobotExperiment(exp_params, {})
    
    superpixel_seg = SuperpixelSegmentation({'spatial_radius' : 1.5, 'volume_threshold' : 150 })
    superpixel_seg_syn = SuperpixelSegmentationSyn({'spatial_radius' : 1.5, 'volume_threshold' : 150, 'convnet_file' : convnet_file })
    superdeep_seg = SuperDeepSegmentation({'spatial_radius' : 1.5, 'volume_threshold' : 150, 'convnet_file' : convnet_file, 'convnet_file_basins' : convnet_file_basins  ,'type_weights' : type_weights })
    pixel_seg = PixelSegmentation({})
    pixel_grad_seg = PixelGradSegmentation({})
    pixel_grad_sobel_seg = PixelGradSegmentationSobel({})
    pixel_grad_enh_seg = PixelGradSegmentationEnhancement({'spatial_radius' : 1.5, 'volume_threshold' : 150 })
    pixel_grad_enh_conv_seg = PixelGradSegmentationEnhancementConvNet({'spatial_radius' : 1.5, 'volume_threshold' : 150, 'convnet_file' : convnet_file, 'convnet_file_basins' : convnet_file_basins})
    gc_seg = GCSegmentation({ "beta" : 7.0 })
    
    #for robot in [superpixel_robot,pixel_robot,geodesic_robot]:
#    for robot in [pixel_robot]:
#    for robot in [superpixel_robot]:
    for robot in [geodesic_robot]:
        #for segmentation in [superpixel_seg,superdeep_seg]:
        for segmentation in [superpixel_seg]:

            print("Running experiment.")
            stats = robot.run(segmentation)
            
            print("Results for {0}, {1}".format(robot,segmentation))
            for i in range(exp_params['number_iterations']):
                print("Iteration {0}, f-score: {1:.3f}".format(i+1, stats[i]['fscore']))
    print time.time() - start_time, "seconds"

if __name__ == '__main__':
    main()
