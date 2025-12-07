#!/usr/bin/python2.7
import matplotlib
matplotlib.use('Agg')
import experiments_final
import argparse
import os
import pyift
import pylab
import numpy
from skimage import io
from pyslic import slic
import time

class SuperpixelMerging:
    def __init__(self, params):
        self.vol_threshold = params['vol_threshold']
        self.spatial_radius = params['spatial_radius']
        self.area_threshold = params['area_threshold']
        self.alpha = params['alpha']
        self.bins_per_band = params['bins_per_band']
        self.colorspace = { 'ycbcr' : 0, 'rgb' : 1, 'lab' : 4}[params['colorspace']]
        self.area_threshold = params['area_threshold']
    
    def segment(self, image, nregions):
        starting = time.time()
        
        adj1 = pyift.iftCircular(self.spatial_radius)
        adj2 = pyift.iftCircular(1.0)
        
        basins = pyift.iftImageBasins(image, adj1)
        marker = pyift.iftVolumeClose(basins, self.vol_threshold)
        
        wg_label_image = pyift.iftWaterGray(basins, marker, adj1)
        
        dataset = pyift.iftSupervoxelsToSelectiveSearchDataset(image, wg_label_image, self.bins_per_band, self.colorspace)
        dataset.set_alpha(self.alpha)
        
        rh = pyift.iftCreateRegionHierarchySelectiveSearch(dataset, adj2)
        
        ending = time.time()
        
        print(" superpixel merging setup time: {0}s".format(ending - starting))
        
        starting = time.time()
        label_image = pyift.iftFlattenRegionHierarchy(rh, nregions)

        dataset = pyift.iftSupervoxelsToSelectiveSearchDataset(image, label_image, self.bins_per_band, self.colorspace)
        dataset.set_alpha(self.alpha)
        
        ending = time.time()
        
        print(" superpixel merging flattening time: {0}s".format(ending - starting))
        
        
        return pyift.iftEliminateRegionsByArea(dataset, adj2, self.area_threshold)
    
    def __repr__(self):
        return 'superpixel_merging'

class WatershedRegionClosing: 
    def __init__(self, params):
        self.spatial_radius = params['spatial_radius']
    
    def segment(self, image, vol_threshold):
        starting = time.time()
        
        adj1 = pyift.iftCircular(self.spatial_radius)
        basins = pyift.iftImageBasins(image, adj1)
        
        ending = time.time()
        
        print(" watershed closing setup time: {0}s".format(ending - starting))
        
        starting = time.time()
        marker = pyift.iftVolumeClose(basins, vol_threshold)
        
        ret =  pyift.iftWaterGray(basins, marker, adj1)
        ending = time.time()
        
        print(" watershed closing additional time: {0}s".format(ending - starting))
        
        return ret
    
    def __repr__(self):
        return 'watershed_closing'

class GraphBasedSegmentation:
    def __init__(self, params):
        self.sigma = params['sigma']
        self.min = params['min']
        self.app_path = params['app_path']
        
        tmp_path = params['tmp_path']
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        
        self.tmp_input_path = os.path.join(tmp_path,'tmp_input.ppm')
        self.tmp_output_path = os.path.join(tmp_path,'tmp_output.ppm')
    
    def segment(self, image, k):
        pyift.iftWriteImageP6(image,self.tmp_input_path)
        os.system('{0} {1} {2} {3} {4} {5}'.format(self.app_path, self.sigma, k, self.min, self.tmp_input_path, self.tmp_output_path))
        
        colored_label = pyift.iftReadImageP6(self.tmp_output_path)
        return pyift.relabel_color_image(colored_label)
    
    def __repr__(self):
        return 'graph_based_seg'
    
class SLICSegmentation:
    def __init__(self, params):
        self.compactness = params['compactness']
        self.app_path = params['app_path']
        
        tmp_path = params['tmp_path']
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        
        self.tmp_input_path = os.path.join(tmp_path,'tmp_input.ppm')
        self.tmp_output_path = os.path.join(tmp_path,'tmp_output.pgm')
    
    def segment(self, image, nregions):
        pyift.iftWriteImageP6(image, self.tmp_input_path)
        
        starting = time.time()
        im = numpy.array(io.imread(self.tmp_input_path))
        region_labels = slic.slic_n(im, nregions, self.compactness)
        ending = time.time()
        
        print(" slic total time: {0}s".format(ending - starting))
        
        label_image = pyift.iftCreateImage(image.xsize, image.ysize, image.zsize)
        pyift.setIftImage(label_image, list(region_labels.reshape(-1) + 1))
        
        return label_image
    
    def __repr__(self):
        return 'slic_seg'

class MergingExperiment:
    def __init__(self, dataset_path, error_tolerance, seg_results):
        self.current_path = os.path.expanduser(os.path.join(dataset_path, 'orig'))
        self.current_images = [f for f in os.listdir(self.current_path) if f.endswith(".ppm")]
        self.error_tolerance = error_tolerance
        self.under_seg_tolerance = 0.05
        
        self.seg_results = seg_results
    
    def run(self, method, size_parameters_list, metric = 'boundary_recall'):
        nregions = [0.0] * len(size_parameters_list)
        br_scores = [0.0] * len(size_parameters_list)
        us_scores = [0.0] * len(size_parameters_list)

        rgb_color = pyift.iftColor()
        rgb_color.set_values((255,0,0))
        ycbcr_color = pyift.iftRGBtoYCbCr(rgb_color)
        adj2 = pyift.iftCircular(1.0)
        adj3 = pyift.iftCircular(1.0)

        for image_path in self.current_images:
            print("Image: {0}".format(image_path))
            
            image = pyift.iftReadImageP6(os.path.join(self.current_path, image_path))
            gt_image = pyift.iftReadImageP5(os.path.join(self.current_path, "../labels/" + image_path[:-3] + "pgm"))

            for j,parameter in enumerate(size_parameters_list):
                label_image = method.segment(image, parameter)
                
                if self.seg_results:
                    overlay = pyift.iftCopyImage(image)
                    pyift.iftDrawBorders(overlay, label_image, adj2, ycbcr_color, adj3 )
                    pyift.iftWriteImageP6(overlay,os.path.join(self.seg_results, '{0}_{1}_{2}.ppm'.format(str(method),image_path[:-4],parameter)))

                #FIXME
                print('total regions {0}'.format(pyift.iftMaximumValue(label_image)))
                nregions[j] += pyift.iftMaximumValue(label_image)
                
                br_scores[j] += self.boundary_recall(gt_image, label_image)
                us_scores[j] += self.under_seg_error(gt_image, label_image)
        
        nregions = [nr/len(self.current_images) for nr in nregions]
        br_scores = [s/len(self.current_images) for s in br_scores]
        us_scores = [s/len(self.current_images) for s in us_scores]
            
        return nregions, br_scores, us_scores
    
    def boundary_recall(self, gt_image, label_image):
        adj = pyift.iftCircular(self.error_tolerance)
        return pyift.iftBoundaryRecall(gt_image, label_image, adj)
    
    def under_seg_error(self, gt_image, label_image):
        return pyift.iftUnderSegmentationMin(gt_image, label_image)

class MergingBerkeleyExperiment:
    def __init__(self, dataset_path, error_tolerance, seg_results):
        self.current_path = os.path.expanduser(os.path.join(dataset_path, 'orig'))
        self.current_images = [f for f in os.listdir(self.current_path) if f.endswith(".ppm")]
        self.error_tolerance = error_tolerance
        self.under_seg_tolerance = 0.05
        
        self.seg_results = seg_results
    
    def run(self, method, size_parameters_list):
        nregions = [0.0] * len(size_parameters_list)
        br_scores = [0.0] * len(size_parameters_list)
        us_scores = [0.0] * len(size_parameters_list)

        rgb_color = pyift.iftColor()
        rgb_color.set_values((255, 0, 0))
        ycbcr_color = pyift.iftRGBtoYCbCr(rgb_color)
        adj2 = pyift.iftCircular(1.0)
        adj3 = pyift.iftCircular(1.0)

        for image_path in self.current_images:
            print("Image: {0}".format(image_path))
            
            image = pyift.iftReadImageP6(os.path.join(self.current_path, image_path))

            gt_images_seg = [ pyift.iftReadImageP5(os.path.join(self.current_path, '../segment/', f)) for f in os.listdir(os.path.join(self.current_path, "../segment/")) if f.startswith(image_path[:-4] + '_') ]
            
            gt_images_bord = [ pyift.iftReadImageP5(os.path.join(self.current_path, '../border/', f)) for f in os.listdir(os.path.join(self.current_path, "../segment/")) if f.startswith(image_path[:-4] + '_') ]    
            gt_image_final = gt_images_bord[0]
            for img in gt_images_bord:
                gt_image_final = pyift.iftOr(gt_image_final, img)

            for j, parameter in enumerate(size_parameters_list):
                label_image = method.segment(image, parameter)
                
                if self.seg_results:
                    overlay = pyift.iftCopyImage(image)
                    pyift.iftDrawBorders(overlay, label_image, adj2, ycbcr_color, adj3)
                    pyift.iftWriteImageP6(overlay, os.path.join(self.seg_results, '{0}_{1}_{2}.ppm'.format(str(method), image_path[:-4], parameter)))
                
                nregions[j] += pyift.iftMaximumValue(label_image)
		
                br_scores[j] += self.boundary_recall(gt_image_final, label_image)
                us_scores[j] += self.under_seg_error(gt_images_seg, label_image)
        
        nregions = [nr / len(self.current_images) for nr in nregions]
        br_scores = [s / len(self.current_images) for s in br_scores]
        us_scores = [s / len(self.current_images) for s in us_scores]
            
        return nregions, br_scores, us_scores
    
    def boundary_recall(self, gt_image, label_image):
        adj = pyift.iftCircular(self.error_tolerance)
        return pyift.iftBoundaryRecallFromBImage(gt_image, label_image, adj)
    
    def under_seg_error(self, gt_images, label_image):
        scores = numpy.array([ pyift.iftUnderSegmentationMin(gt_image, label_image) for gt_image in gt_images])
        return scores.mean()


def main():
    parser = argparse.ArgumentParser(description="Experiments for region merging")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset")
    parser.add_argument('output_path', type=str, help='Path to the output folder')
    parser.add_argument('--berkeley', action = 'store_true', default = False, help = 'Toggles the dataset format to the format adopted by the Berkeley database')

    args = parser.parse_args()

    error_tolerance = 2.0
    
    if args.berkeley:
        merging_experiment = MergingBerkeleyExperiment(args.dataset_path, error_tolerance, None)
    else:
        merging_experiment = MergingExperiment(args.dataset_path, error_tolerance, None)
    
    #Superpixel merging
    parameters_merging = { 
                          'vol_threshold' : 150,
                           'spatial_radius' : 3.0,
                           'alpha' : [0.2, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0],
                           'bins_per_band' : 4,
                           'colorspace' : 'lab',
                           'area_threshold' : 20
                         }
    superpixel_merging = SuperpixelMerging(parameters_merging)
    
    #Watershed volume closing
    parameters_closing = {
                          'spatial_radius' : 3.0,
                          }
    watershed_closing = WatershedRegionClosing(parameters_closing)
    
    #Graph based segmentation
    parameters_gbs =   { 
                       'sigma' : 0.5,
                       'min' : 20,
                       'tmp_path' : os.path.join(args.output_path,'tmp'),
                       'app_path' : 'gbs/segment'
                       }
    graph_based_seg = GraphBasedSegmentation(parameters_gbs)
    
    #SLIC
    parameters_slic =   { 
                       'compactness' : 10,
                       'tmp_path' : os.path.join(args.output_path,'tmp')
                       }
    slic_seg = SLICSegmentation(parameters_slic)
    
    methods = [superpixel_merging, watershed_closing, graph_based_seg, slic_seg]
    ranges = [ [int(x) for x in numpy.linspace(20,500,5)], [int(x) for x in numpy.linspace(500,100000,5)], [int(x) for x in numpy.linspace(200,10000,5)], [int(x) for x in numpy.linspace(20,500,5)]]

    for i in range(len(methods)):
        print('Testing: {0}'.format(str(methods[i])))
        nregions, br_scores, us_scores = merging_experiment.run(methods[i], ranges[i])
        
        pylab.subplot(1,2,1)
        pylab.plot(nregions, br_scores, '-o', label = '{0}_br'.format(methods[i]))
        
        pylab.subplot(1,2,2)
        pylab.plot(nregions, us_scores, '-o', label = '{0}_us'.format(methods[i]))
        
        numpy.save(os.path.join(args.output_path,'merging_nregions_{0}.npy'.format(methods[i])),nregions)
        numpy.save(os.path.join(args.output_path,'merging_br_scores_{0}'.format(methods[i])),br_scores)
        numpy.save(os.path.join(args.output_path,'merging_us_scores_{0}'.format(methods[i])),us_scores)
        
    pylab.legend(loc = 2)
    pylab.savefig(os.path.join(args.output_path,'merging.pdf'))

if __name__ == '__main__':
    main()
