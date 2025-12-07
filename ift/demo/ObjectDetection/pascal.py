#!/usr/bin/python2.7
import os
import numpy
from lxml import etree
import skimage
from skimage import transform
from skimage import io
from skimage import draw
from sklearn import metrics
import random

class Rectangle:
    def __init__(self, xmax, ymax, xmin, ymin):
        """Create a rectangle given coordinates
        
        Parameters
        ----------
        xmax: maximum x coordinate 
        ymax: maximum y coordinate
        xmin: minimum x coordinate 
        ymin: minimum y coordinate
        
        """
        if xmax < xmin or ymax < ymin:
            raise Exception("Invalid Rectangle")
        
        self.xmax = xmax
        self.xmin = xmin
        self.ymin = ymin
        self.ymax = ymax
        
    def intersection_area(self, other):
        """Return the intersection area between two rectangles"""
        x_overlap = max(0, min(self.xmax,other.xmax) - max(self.xmin,other.xmin) + 1)
        y_overlap = max(0, min(self.ymax,other.ymax) - max(self.ymin,other.ymin) + 1)
        
        return x_overlap*y_overlap
    
    def intersection_score(self, other):
        """Return the intersection score between two rectangles"""
        intersection_area = self.intersection_area(other)
        return float(intersection_area)/(self.area() + other.area() - intersection_area)
    
    def area(self):
        """Return the area of the rectangle"""
        return self.width()*self.height()
    
    def center(self):
        """Return a pair representing the center of the rectangle"""
        return ( (self.xmax + self.xmin)/2, (self.ymax + self.ymin)/2 )
    
    def width(self):
        """Return the width of the rectangle"""
        return self.xmax - self.xmin + 1
    
    def height(self):
        """Return the height of the rectangle"""
        return self.ymax - self.ymin + 1
    
    def __add__(self, offsets):
        """Offset the rectangle by a pair of coordinates"""
        offset_x, offset_y = offsets
        return Rectangle( self.xmax + offset_x, self.ymax + offset_y, self.xmin + offset_x, self.ymin + offset_y)
        
    def contains(self, other):
        """Return whether the rectangle contains another rectangle"""
        return numpy.allclose(self.intersection_area(other),other.area())
    
    def __repr__(self):
        """Return the string representation of this rectangle"""
        return '{0}-{1}\n{2}-{3}'.format((self.xmin,self.ymin),(self.xmax,self.ymin),(self.xmin,self.ymax),(self.xmax,self.ymax))
    
    def __eq__(self, other):
        """Return whether this rectangle is equal other object"""
        return  self.xmax == other.xmax and \
                self.ymax == other.ymax and \
                self.xmin == other.xmin and \
                self.ymin == other.ymin 
                
    def __hash__(self):
        return self.area() + self.xmin ^ self.ymin
    
class Frame(Rectangle):
    """Rectangle contained in an image. May be used to represent objects of interest"""
    def __init__(self,xmax, ymax, xmin, ymin, image, img_name, class_name, confidence = 1.0):
        """Create a frame.
        
        Parameters
        ----------
        xmax: maximum x coordinate 
        ymax: maximum y coordinate
        xmin: minimum x coordinate 
        ymin: minimum y coordinate
        image: image that contains this frame (numpy.array)
        img_name: name of the image that contains this frame (string, no extension)
        
        class_name: real or attributed class of this frame. None if frame is not classified. 
        confidence: confidence on the given class_name
        """
        Rectangle.__init__(self, xmax, ymax, xmin, ymin)
        
        self.img_name = img_name
        self.class_name = class_name
        self.confidence = confidence
        
        self.frame = image[self.ymin:self.ymax+1,self.xmin:self.xmax+1]
        
    def reframe_by_windows(self, window_sizes, image):
        """Return a frame by reframing this frame inside one of the available window_sizes (if possible).
        Return None if no window size suffices.
        
        Parameters
        ----------
        window_sizes: list of window_sizes (pairs)
        image: image that contains this frame (ndarray)
        
        Return
        ------
        frame or None
        """
        obj_area = self.area()
        obj_centerx,obj_centery = self.center()
        
        image_rect = Rectangle(image.shape[1] - 1, image.shape[0] -1, 0, 0)
        
        #Chooses the best window size to frame this object
        window_size = None
        best_score = float("-inf")
        for w in window_sizes:
            wrect = Rectangle(w[0] -1, w[1] - 1, 0, 0)
            wcenter = wrect.center()
            
            offset_x = obj_centerx - wcenter[0]
            offset_y = obj_centery - wcenter[1]
            
            wrect = wrect + (offset_x, offset_y)
            score = self.intersection_score(wrect)
            
            if score > best_score and image_rect.contains(wrect):
                best_score = score
                window_size = w
                
        if window_size == None:
            return None
        
        wrect = Rectangle(window_size[0] -1, window_size[1] - 1, 0, 0)
        wcenter = wrect.center()
        offset_x = obj_centerx - wcenter[0]
        offset_y = obj_centery - wcenter[1]
        
        wrect = wrect + (offset_x, offset_y)
        
        return Frame(wrect.xmax, wrect.ymax, wrect.xmin, wrect.ymin,\
                              image, self.img_name, self.class_name, self.confidence)
        
    def reframe_by_ratios(self, ratios, image):
        """Return a frame by reframing this frame to one of the available aspect ratios. 
        Returns None if no ratio suffices.
        
        Parameters
        ----------
        ratios: list of aspect ratios (width/height)
        image: image that contains this frame (ndarray)
        
        Return
        ------
        frame or None
        """
        frame_ratio = float(self.width())/self.height()
        image_rect = Rectangle(image.shape[1] - 1, image.shape[0] -1, 0, 0)
        
        sorted_ratios = sorted(ratios, key = lambda x: abs(x - frame_ratio))

        for ratio in sorted_ratios:
            if ratio > frame_ratio:
                height = self.height()
                width = height*ratio
                width_inc = int((width - self.width())/2)
                
                rect = Rectangle(self.xmax +width_inc, self.ymax, self.xmin - width_inc, self.ymin)
                if image_rect.contains(rect):
                    return Frame(rect.xmax, rect.ymax, rect.xmin, rect.ymin, image, self.img_name, self.class_name, self.confidence)
            elif ratio < frame_ratio:
                width = self.width()
                height = width/ratio 
                height_inc = int((height - self.height())/2)
                
                rect = Rectangle(self.xmax, self.ymax + height_inc, self.xmin, self.ymin - height_inc)
                if image_rect.contains(rect):
                    return Frame(rect.xmax, rect.ymax, rect.xmin, rect.ymin, image, self.img_name, self.class_name, self.confidence)
            else:
                return Frame(self.xmax, self.ymax, self.xmin, self.ymin, image, self.img_name, self.class_name, self.confidence)

        #No ratio suffices
        return None
                    
    def save(self, path):
        """Save this frame to a file"""
        io.imsave(path, self.frame)
        
    def save_ift(self, path, window_size):
        """Resize this frame to window_size x window_size and saves it to a file (in grayscale)."""
        print("path: {0}",format(path));
        window_size = float(window_size)
        if self.width() <= 1 or self.height() <= 1:
            io.imsave(path, numpy.zeros((window_size, window_size)))
        else:
            resizing_factor = window_size/max(self.frame.shape[0:2])
            frame = transform.rescale(self.frame,resizing_factor,mode = 'nearest')
            
            if frame.shape != (window_size, window_size, 3):
                output_obj = numpy.zeros((window_size,window_size,3))
                initial_coord = ( int((window_size - frame.shape[0])/2), int((window_size - frame.shape[1])/2) )
                final_coord = ( initial_coord[0] + frame.shape[0] , initial_coord[1] + frame.shape[1] )
                output_obj[initial_coord[0]:final_coord[0],initial_coord[1]:final_coord[1]] = frame
                
                frame = output_obj
            
            io.imsave(path, skimage.color.rgb2gray(frame))
        
    def draw(self, image):
        """Draw a red frame inside a copy of image and return it"""
        image = numpy.array(image)
        y_coord = [self.ymin, self.ymin, self.ymax, self.ymax]
        x_coord = [self.xmin, self.xmax, self.xmax, self.xmin]
        
        x_coord = [max(0, min(image.shape[1] -1, c)) for c in x_coord]
        y_coord = [max(0, min(image.shape[0] -1,c)) for c in y_coord]
        
        rr,cc = draw.line(y_coord[0],x_coord[0],y_coord[1],x_coord[1])
        image[rr,cc] = (255,0,0)
        rr,cc = draw.line(y_coord[1],x_coord[1],y_coord[2],x_coord[2])
        image[rr,cc] = (255,0,0)
        rr,cc = draw.line(y_coord[2],x_coord[2],y_coord[3],x_coord[3])
        image[rr,cc] = (255,0,0)
        rr,cc = draw.line(y_coord[3],x_coord[3],y_coord[0],x_coord[0])
        image[rr,cc] = (255,0,0)
        
        return image
    
    def __repr__(self):
        rect =  'Frame coordinates: \n{0}-{1}\n{2}-{3}.\n'.format((self.xmin,self.ymin),(self.xmax,self.ymin),(self.xmin,self.ymax),(self.xmax,self.ymax))
        rect += 'Frame properties: Image {0}. Confidence {1}.\n'.format(self.img_name, self.confidence)
        return rect
    
    def __eq__(self, other):
        return  Rectangle.__eq__(self, other)   and \
                self.img_name == other.img_name and \
                self.class_name == other.class_name and \
                numpy.allclose(self.confidence, other.confidence) 
                #and \numpy.allclose(self.frame, other.frame)
                
class Pascal:
    """Class for managing a PASCAL VOC database"""
    def __init__(self, path):
        """Create an instance given the path to database""" 
        if not os.path.exists(os.path.join(path,'VOC2007/')):
            raise Exception('Invalid path to database.')
            
        self.path = path
        self.image_path = os.path.join(self.path, 'VOC2007/JPEGImages')
        self.annot_path = os.path.join(self.path, 'VOC2007/Annotations')
        
        self.training_list = open(os.path.join(self.path, 'VOC2007/ImageSets/Main/train.txt')).readlines()
        self.training_list = [i.strip() for i in self.training_list]
        
        self.validation_list = open(os.path.join(self.path, 'VOC2007/ImageSets/Main/val.txt')).readlines()
        self.validation_list = [i.strip() for i in self.validation_list]
        
        self.testing_list = open(os.path.join(self.path, 'VOC2007/ImageSets/Main/test.txt')).readlines()
        self.testing_list = [i.strip() for i in self.testing_list]
    
        self.classes = ['cat', 'sofa', 'dog', 'tvmonitor', 'bus', 'train', 'diningtable', 'aeroplane', 'horse', 'motorbike', 
                        'bicycle', 'cow', 'bird', 'chair', 'car', 'sheep', 'boat', 'person', 'pottedplant', 'bottle']
        
    def get_image(self, img_name):
        """Return an image as a numpy.darray given its name (without extension)"""
        return io.imread(os.path.join(self.image_path, img_name + '.jpg'))
    
    def get_annotations(self, img_name):
        """Returns the annotation tree for a image given its name (without extension)"""
        return etree.parse(os.path.join(self.annot_path,img_name + '.xml'))
        
    def get_positive_images(self,image_list, positive_class):
        """Return a list of image names that contain objects of the positive_class"""
        pos_images = set()
        for img_name in image_list:
            annot_tree = self.get_annotations(img_name)
            obj_trees = list(annot_tree.iter('object'))
            for obj in obj_trees:
                if int(obj.find('difficult').text):
                    continue
                
                obj_class = obj.find('name').text
                if obj_class == positive_class:
                    pos_images.add(img_name)
                    
        return list(pos_images)
    
    def get_negative_images(self, image_list, positive_class):
        """Return a list of image names that do not contain objects of the positive_class"""
        positives = set(self.get_positive_images(image_list, positive_class))
        return [n for n in image_list if n not in positives]
        
    def get_positive_frames(self, image_list, positive_class):
        """Return all the positive frames for a given positive_class. 
        
        Parameters
        ----------
        image_list: list of image names from which to extract the positive objects
        positive_class: class of the positive objects
        
        Return
        ------
        list of frames belonging to positive_class
        """
        frames = []
        
        for img_name in image_list:
            image = self.get_image(img_name)
            annot_tree = self.get_annotations(img_name)
    
            obj_trees = list(annot_tree.iter('object'))
            for obj in obj_trees:
                if int(obj.find('difficult').text):
                    continue
                
                obj_class = obj.find('name').text
                if obj_class != positive_class:
                    continue
                
                bndbox_tree = obj.find('bndbox')
                xmax = int(bndbox_tree.find('xmax').text)
                ymax = int(bndbox_tree.find('ymax').text)
                xmin = int(bndbox_tree.find('xmin').text)
                ymin = int(bndbox_tree.find('ymin').text)
                
                frame = Frame(xmax,ymax,xmin,ymin, image, img_name, positive_class)
                frames.append(frame)
                    
        return frames
    
    def get_regions_superpixels(self, image_list, superpixel_volume, regions_per_image, nbins, alpha, verbose = False):
        """Returns frames centered on superpixels obtained by a hybrid watershed
         segmentation.
         
         Important: you need to convert the images to ppm before calling this function.
        
        Parameters
        ----------
        image_list: list of image names from which to extract frames 
        supuperpixel_volume : initial volume of the superpixels (int)
        total_regions: number of superpixels to evaluate (sorted by decreasing area)
        in the image should contain a window.
        Return
        ------
        list of frames
        """
        from pyift import pyift
        
        regions = []
        
        for i, image_name in enumerate(image_list):
            if verbose:
                print("Extracting hybrid superpixels from image {0}/{1}".format(i+1, len(image_list)))
                
            image_path = os.path.join(self.image_path, image_name + '.ppm')
            image = self.get_image(image_name)
            
            rectangle_coords = pyift.get_hybrid_superpixel_windows_full(image_path, superpixel_volume, alpha, nbins)
            frames = [Frame(t[0],t[1],t[2],t[3], image, image_name, None) for t in rectangle_coords]
            frames.sort(key = lambda x: x.area(), reverse = True)
            frames = frames[:regions_per_image]
            
            regions += frames
           
        return regions 
    
    def get_regions_random(self, image_list, regions_per_image):
        """Returns a random number of regions per image"""
        regions = []
        
        for i, image_name in enumerate(image_list):
            image = self.get_image(image_name)
            image_rect = Rectangle(image.shape[1] - 1, image.shape[0] -1, 0, 0)
            
            for _ in range(regions_per_image):
                x1 = random.randint(0, image.shape[1] - 1)
                y1 = random.randint(0, image.shape[0] - 1)
                
                x2 = random.randint(0, image.shape[1] - 1)
                y2 = random.randint(0, image.shape[0] - 1)
                
                frame = Frame(max(x1,x2), max(y1,y2), min(x1,x2), min(y1,y2), image, image_name, None)
                regions.append(frame)
                
        return regions 
    
    def get_regions_kde(self, image_list, positive_class, regions_per_image, verbose = False):
        """Returns a region sampled from a gaussian kernel density estimator"""
        from scipy.stats import gaussian_kde
        
        pos_frames = self.get_positive_frames(self.training_list, positive_class)
        X = numpy.zeros((len(pos_frames), 4))
        for i in range(X.shape[0]):
            f = pos_frames[i]
            image = self.get_image(f.img_name)
            X[i] = [ float(f.width())/image.shape[1],float(f.height())/image.shape[0], float(f.xmin)/image.shape[1], float(f.ymin)/image.shape[0]]
        
        gkde = gaussian_kde(X.T)
        
        regions = []
        for i, image_name in enumerate(image_list):
            image_path = os.path.join(self.image_path, image_name + '.ppm')
            image = self.get_image(image_name)
            
            image_rect = Rectangle(image.shape[1] - 1, image.shape[0] - 1, 0, 0)
            
            image_regions = []
            
            Xsamples = []
            j = regions_per_image
            while len(image_regions) != regions_per_image:
                if j >= regions_per_image:
                    Xsamples = gkde.resample(regions_per_image).T
                    j = 0
                
                x_vec = Xsamples[j].reshape(-1)
                j += 1
                
                xmin = int(x_vec[2] * image.shape[1])
                ymin = int(x_vec[3] * image.shape[0])
                xmax = int(xmin + (x_vec[0] * image.shape[1]) - 1)
                ymax = int(ymin + (x_vec[1] * image.shape[0]) - 1)
                
                try:
                    frame = Frame(xmax, ymax, xmin, ymin, image, image_name, None)
                    if image_rect.contains(frame):
                        image_regions.append(frame)
                except Exception:
                    pass
            
            regions += image_regions

        return regions

    def evaluate_detection(self, image_list, detected_regions):
        """
        Return the precision, recall and thresholds for the detected_regions in image_list
        
        Parameters
        ----------
        image_list: list of image names containing all the regions in consideration
        detected_regions: list of detected regions (with class and confidence assignment)
        
        Return:
        a triple representing a precision/recall curve:
        thresholds: array of confidence thresholds
        precision: array containing the precision for regions above certain threshold
        recall: array containing the recalls for regions above certain threshold
        
        """
        detected_regions = list(detected_regions)
        if not detected_regions:
            raise Exception('No detected regions')
        
        image_list_set = set(image_list)
        if not image_list_set:
            raise Exception('No positive images')
        
        positive_class = detected_regions[0].class_name 
        
        for det_region in detected_regions:
            if det_region.class_name != positive_class:
                raise Exception('A detected region is not a positive sample')
            if det_region.img_name not in image_list_set:
                raise Exception('A detected region does not belong to an image in the list')
         
        confidences = numpy.array([float(d.confidence) for d in detected_regions])
        if not numpy.allclose(confidences.max(), confidences.min()):
            confidences = (confidences - confidences.min())/(confidences.max() - confidences.min())
        else:
            confidences = numpy.ones(len(confidences))
         
        thresholds = numpy.sort(list(set(confidences)))
         
        precision = numpy.zeros(len(thresholds) + 1)
        recall = numpy.zeros(len(thresholds) + 1)
         
        precision[-1] = 1.
        recall[-1] = 0.
        
        all_regions = list(self.get_positive_frames(image_list, positive_class))
        if not all_regions:
            raise Exception('No positive frames')
        
        for t,threshold in enumerate(thresholds):
            regions_above_threshold = [det_r for i,det_r in enumerate(detected_regions) if confidences[i] >= threshold]
            
            already_detected = numpy.zeros(len(all_regions), dtype = bool)
            
            for det_r in regions_above_threshold:
                for i,cor_r in enumerate(all_regions):
                    if cor_r.img_name == det_r.img_name and cor_r.intersection_score(det_r) >= 0.5:
                        if not already_detected[i]:
                            already_detected[i] = True
                            break
            
            tp = already_detected.sum()            
            fp = len(regions_above_threshold) - tp
            fn = len(already_detected) - tp
            
            if tp + fp == 0:
                raise Exception('No frames above threshold')
            if tp + fn == 0:
                raise Exception('No positive frames')
             
            precision[t] = float(tp)/(tp + fp)
            recall[t] = float(tp)/(tp + fn)
             
        return precision, recall, thresholds
    
    def evaluate_recall(self, positive_regions, regions, output_path = False, plot = False):
        """Evaluate recall for an image_list given the detected regions
        
        Parameters
        ----------
        positive_regions: list of frames containing the positive regions
        regions: list of frames containing the detected regions
        plot: whether to plot the sizes of the undetected regions 
        
        Return:
        recall (float)
        """
        if not positive_regions:
            raise Exception('No positive frames')
        
        if output_path:        
            if not os.path.exists(os.path.join(output_path,'found')):
                os.makedirs(os.path.join(output_path,'found'))
            if not os.path.exists(os.path.join(output_path,'not_found')):
                os.makedirs(os.path.join(output_path,'not_found'))
        
        image_to_regions = { rs.img_name : [] for rs in regions }
        for rs in regions:
            image_to_regions[rs.img_name].append(rs)
        
        found = numpy.zeros(len(positive_regions), dtype = bool)
        max_iscore = numpy.zeros(len(positive_regions))
        for i,r in enumerate(positive_regions):
            if r.img_name in image_to_regions:
                det_regions = image_to_regions[r.img_name]
                
                for rs in det_regions:
                    iscore = rs.intersection_score(r)
                    
                    if max_iscore[i] < iscore:
                        max_iscore[i] = iscore
                    
                    if  iscore >= 0.5:
                        found[i] = True
                    
                        if output_path:
                            output_prefix = os.path.join(output_path,'found/' + rs.img_name + '_{0}.jpg'.format(i+1))
                            io.imsave(output_prefix, rs.draw(self.get_image(rs.img_name)) )
        
        if plot:
            import pylab
            found_regions = [r for i,r in enumerate(positive_regions) if found[i]]
            not_found_regions = [r for i,r in enumerate(positive_regions) if not found[i]]
            
            pylab.plot([r.width() for r in found_regions], [r.height() for r in found_regions],'bo')
            pylab.plot([r.width() for r in not_found_regions], [r.height() for r in not_found_regions],'ro')
            
            pylab.show()
            
            print("Average Best Overlap: {0} +-{1}".format(max_iscore.mean(),max_iscore.std()))
            
            if output_path:
               for i,r in enumerate(not_found_regions):
                   output_prefix = os.path.join(output_path,'not_found/' + r.img_name + '_{0}.jpg'.format(i+1))
                   io.imsave(output_prefix, r.draw(self.get_image(r.img_name)) )
        
        tp = found.sum()
        fn = len(found) - tp
            
        return float(tp)/(tp + fn)
    
def create_trainval(pascal, positive_class, negative_samples):
    """Create a training and validation dataset for a given positive class
    
    Parameters
    ----------
    pascal: instance of the class Pascal 
    positive_class: name of the positive class
    negative_samples: number of windows per image
    
    Return
    tuple pos_train, neg_train, pos_val, neg_val. Each element is a list of frames.
    """
            
    #Positive frames
    pos_train = pascal.get_positive_frames(pascal.training_list, positive_class)
    pos_val = pascal.get_positive_frames(pascal.validation_list, positive_class)
 
    #Negative frames
    windows_per_image = negative_samples/len(pascal.training_list)
    if windows_per_image < 1:
        images = list(pascal.training_list) 
        random.shuffle(images)
        neg_train_c = pascal.get_regions_kde(images[:negative_samples], positive_class, 1)
    else: 
        neg_train_c = pascal.get_regions_kde(pascal.training_list, positive_class, windows_per_image)
    
    windows_per_image = negative_samples/len(pascal.validation_list)
    if windows_per_image < 1:
        images = list(pascal.validation_list)
        random.shuffle(images)
        neg_val_c = pascal.get_regions_kde(images[:negative_samples], positive_class, 1)
    else:
        neg_val_c = pascal.get_regions_kde(pascal.validation_list, positive_class, windows_per_image)

    neg_train = []
    for i,ri in enumerate(neg_train_c):
        invalid = False
        
        for j,rj in enumerate(pos_train):
            if ri.img_name == rj.img_name and ri.intersection_score(rj) >= 0.5:
                invalid = True
        
        if not invalid:
            neg_train.append(ri)
    
    neg_val = []
    for i,ri in enumerate(neg_val_c):
        invalid = False
        
        for j,rj in enumerate(pos_val):
            if ri.img_name == rj.img_name and ri.intersection_score(rj) >= 0.5:
                invalid = True
        
        if not invalid:
            neg_val.append(ri)
            
    return pos_train, neg_train, pos_val, neg_val
            
def create_ift_trainval(pascal, positive_class, negative_samples, output_path, ratios, pgm_output_size = 200):
    """
    Create a libift-compatible training and validation dataset for a given positive class
    
    Parameters
    ----------
    pascal: instance of the class Pascal 
    positive_class: name of the positive class
    negative_samples: number negative samples
    output_path: path in which the dataset will be stored
    pgm_output_size: side of the square ppm image used by the ift convolutional network
    """
    dirs = ['train', 'val']
    for dir in dirs:
        if not os.path.exists(os.path.join(output_path,dir)):
            os.makedirs(os.path.join(output_path,dir))
    
    pos_train, neg_train, pos_val, neg_val = create_trainval(pascal, positive_class, negative_samples)
    
    for i,r in enumerate(pos_train):
        image = pascal.get_image(r.img_name)
        r = r.reframe_by_ratios(ratios, image)
        if r:
            r.save_ift(os.path.join(output_path,'train/{0:06d}_{1:07d}.pgm').format(1,i+1), pgm_output_size)
            
    for i,r in enumerate(pos_val):
        image = pascal.get_image(r.img_name)
        r = r.reframe_by_ratios(ratios, image)
        if r:
            r.save_ift(os.path.join(output_path,'val/{0:06d}_{1:07d}.pgm').format(1,i+1), pgm_output_size)
    
    for i,r in enumerate(neg_train):
        image = pascal.get_image(r.img_name)
        r = r.reframe_by_ratios(ratios, image)
        if r:
            r.save_ift(os.path.join(output_path,'train/{0:06d}_{1:07d}.pgm').format(2,i+1), pgm_output_size)
        
    for i,r in enumerate(neg_val):
        image = pascal.get_image(r.img_name)
        r = r.reframe_by_ratios(ratios, image)
        if r:
            r.save_ift(os.path.join(output_path,'val/{0:06d}_{1:07d}.pgm').format(2,i+1), pgm_output_size)
        
def create_bow_trainval(pascal, positive_class, negative_samples, output_path):
    dirs = ['bow_train','bow_val']
    for dir in dirs:
        if not os.path.exists(os.path.join(output_path,dir)):
            dd = os.path.join(output_path,dir) 
            os.makedirs(dd)
            os.makedirs(os.path.join(dd,'positive'))
            os.makedirs(os.path.join(dd,'negative'))
            
    pos_train, neg_train, pos_val, neg_val = create_trainval(pascal, positive_class, negative_samples)
    
    for i,r in enumerate(pos_train):
        r.save(os.path.join(output_path,'bow_train/positive/{0}.png').format(i+1))
        
    for i,r in enumerate(pos_val):
        r.save(os.path.join(output_path,'bow_val/positive/{0}.png').format(i+1))
        
    for i,r in enumerate(neg_train):
        r.save(os.path.join(output_path,'bow_train/negative/{0}.png').format(i+1))
        
    for i,r in enumerate(neg_val):
        r.save(os.path.join(output_path,'bow_val/negative/{0}.png').format(i+1))
        
def create_ift_arch_training(pascal, output_path, positive_classes, pgm_output_sizes):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for output_size in pgm_output_sizes:
        if not os.path.exists(os.path.join(output_path,str(output_size))):
            os.makedirs(os.path.join(output_path,str(output_size)))
    
    for j,positive_class in enumerate(positive_classes):
        frames = pascal.get_positive_frames(pascal.get_positive_images(pascal.training_list + pascal.validation_list + pascal.testing_list, positive_class),positive_class)

        i = 0
        for frame in frames:
            image = pascal.get_image(frame.img_name)
            frame = frame.reframe_by_ratios([1.0], image)
            if frame:
                for output_size in pgm_output_sizes:
                    frame.save_ift(os.path.join(output_path,'{0}/{1:06d}_{2:06d}.pgm').format(output_size,j+1,i+1), output_size)
                i += 1

