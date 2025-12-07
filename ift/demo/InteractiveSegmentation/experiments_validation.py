#!/usr/bin/python2.7
import experiments_final as ef

import argparse
from sklearn import metrics
from itertools import product
from functools import partial
from multiprocessing import Pool   
import numpy
import os
import os

def write_results(output_file, params, results):
    sorted_results_index = numpy.argsort(-results)
    
    f = open(output_file, 'w')
    for i in sorted_results_index:
        f.write('Parameters: {0}\n'.format(params[i]))
        f.write('Auc: {0}\n\n'.format(results[i]))
        
    f.close()
    
def ift_pixel_validation_func(params, exp_params):
    geodesic_robot= ef.GeodesicRobotExperiment(exp_params, {})
    
    pixel_seg = ef.PixelSegmentation(params)
    
    stats = geodesic_robot.run(pixel_seg)
    fscores = [ s['fscore'] for s in stats]
    
    return metrics.auc(numpy.linspace(0.,1.,len(fscores)),fscores)
    
def ift_pixel_validation_experiment(exp_params, output_file):
    colorspace_code = { 'ycbcr' : 0, 'rgb' : 1, 'lab' : 4}    
    
    params = []
    
    c_spatial_radius = [ 3.0 ]
   
    #YCbCr and LAB
    c_alpha = [ [0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 1.0, 1.0] ]
    c_colorspace = [ colorspace_code['ycbcr'], colorspace_code['lab'] ]
    combinations = list(product(c_spatial_radius, c_alpha, c_colorspace))
    params += [ { 'spatial_radius' : tuple[0], 'alpha' : tuple[1], 'colorspace' : tuple[2]  } for tuple in combinations]
    
    #RGB
    c_alpha = [ [1.0, 1.0, 1.0] ]
    c_colorspace = [ colorspace_code['rgb'] ]
    
    combinations = list(product(c_spatial_radius, c_alpha, c_colorspace))
    params += [ { 'spatial_radius' : tuple[0], 'alpha' : tuple[1], 'colorspace' : tuple[2]  } for tuple in combinations]
    
    print(len(params))
    
    f = partial(ift_pixel_validation_func, exp_params = exp_params)
    
    pool = Pool()
    results = numpy.array(pool.map(f, params))
    
    write_results(output_file, params, results)
    
def gc_pixel_validation_func(params, exp_params):
    geodesic_robot= ef.GeodesicRobotExperiment(exp_params, {})
    
    gc_pixel_seg = ef.GCPixelSegmentation(params)
    
    stats = geodesic_robot.run(gc_pixel_seg)
    fscores = [ s['fscore'] for s in stats]
    
    return metrics.auc(numpy.linspace(0.,1.,len(fscores)),fscores)

def gc_pixel_validation_experiment(exp_params, output_file):
    colorspace_code = { 'ycbcr' : 0, 'rgb' : 1, 'lab' : 4}    
    
    params = []
    
    c_beta = [1,10,20,30]
    c_spatial_radius = [3.0 ]
    
    #YCbCr and LAB
    c_alpha = [ [0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 1.0, 1.0]]
    c_colorspace = [ colorspace_code['ycbcr'], colorspace_code['lab'] ]
    
    combinations = list(product(c_spatial_radius, c_alpha, c_colorspace, c_beta))
    params += [ { 'spatial_radius' : tuple[0], 'alpha' : tuple[1], 'colorspace' : tuple[2], 'beta' : tuple[3]} for tuple in combinations]
    
    #RGB
    c_alpha = [ [1.0, 1.0, 1.0] ]
    c_colorspace = [ colorspace_code['rgb'] ]
     
    combinations = list(product(c_spatial_radius, c_alpha, c_colorspace, c_beta))
    params += [ { 'spatial_radius' : tuple[0], 'alpha' : tuple[1], 'colorspace' : tuple[2], 'beta' : tuple[3]} for tuple in combinations]
    
    print(len(params))
    
    f = partial(gc_pixel_validation_func, exp_params = exp_params)
    
    pool = Pool()
    results = numpy.array(pool.map(f, params))
    
    write_results(output_file, params, results)
    
def normalize_color_weights(colorspace, alpha1, alpha2, alpha3):
    if colorspace == 'ycbcr' or colorspace == 'rgb':
        maxdist = float(alpha1*255**2 + alpha2*255**2 + alpha3*255**2)
    elif colorspace == 'lab':
        maxdist = float(alpha1*100**2 + alpha2*200**2 + alpha3*200**2)
    
    return [alpha1/maxdist, alpha2/maxdist, alpha3/maxdist]
    
def ift_super_validation_func(params, exp_params):
    geodesic_robot= ef.GeodesicRobotExperiment(exp_params, {})
    
    super_seg = ef.SuperpixelSegmentation(params)
    
    stats = geodesic_robot.run(super_seg)
    fscores = [ s['fscore'] for s in stats]
    
    return metrics.auc(numpy.linspace(0.,1.,len(fscores)),fscores)    
    
def ift_super_validation_experiment(exp_params, output_file):
    """alpha - list containing 7 elements weighting: mean band1, mean band2, mean band3, relative area, color histogram intersection, lbp histogram, rectangularity"""
    colorspace_code = { 'ycbcr' : 0, 'rgb' : 1, 'lab' : 4}    
     
    params = []
     
    c_spatial_radius = [ 3.0 ]
    c_volume_threshold = [ 10, 50, 100, 400]
    c_bins_per_band = [ 4, 8]
    
    c_colorspace = [ colorspace_code['ycbcr'], colorspace_code['lab'] ]
     
    color_alphas = [[0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 1.0, 1.0]]
    c_alpha = []
    for a in color_alphas:
        norm_weight = normalize_color_weights('ycbcr', a[0], a[1], a[2])
          
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
        
        norm_weight = normalize_color_weights('lab', a[0], a[1], a[2])
          
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
    c_alpha.append( [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ])   
      
    combinations = list(product(c_spatial_radius, c_alpha, c_colorspace, c_volume_threshold, c_bins_per_band))
    params += [ { 'spatial_radius' : tuple[0], 'alpha' : tuple[1], 
                'colorspace' : tuple[2], 'volume_threshold' : tuple[3],
                'bins_per_band' : tuple[4]  } for tuple in combinations]
    #RGB
    c_colorspace = [ colorspace_code['rgb' ]]
     
    c_alpha = []
    norm_weight = normalize_color_weights('rgb',1.0, 1.0,1.0)
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
    c_alpha.append( [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ])    
      
    combinations = list(product(c_spatial_radius, c_alpha, c_colorspace, c_volume_threshold, c_bins_per_band))
    params += [ { 'spatial_radius' : tuple[0], 'alpha' : tuple[1], 
                'colorspace' : tuple[2], 'volume_threshold' : tuple[3],
                'bins_per_band' : tuple[4]  } for tuple in combinations]
      
    print(len(params))
      
    f = partial(ift_super_validation_func, exp_params = exp_params)
      
    pool = Pool()
    results = numpy.array(pool.map(f, params))
      
    write_results(output_file, params, results)
    
def gc_super_validation_func(params, exp_params):
    geodesic_robot= ef.GeodesicRobotExperiment(exp_params, {})
    
    gc_super_seg = ef.GCSuperSegmentation(params)
    
    stats = geodesic_robot.run(gc_super_seg)
    fscores = [ s['fscore'] for s in stats]
    
    return metrics.auc(numpy.linspace(0.,1.,len(fscores)),fscores)    
    
def gc_super_validation_experiment(exp_params, output_file):
    colorspace_code = { 'ycbcr' : 0, 'rgb' : 1, 'lab' : 4}    
    
    params = []
    
    c_spatial_radius = [ 3.0 ]
    c_volume_threshold = [ 10, 50, 100, 400]
    c_bins_per_band = [ 4, 8]
    c_beta = [1,10,20,30]
    
    #YCbCr and LAB
    c_colorspace = [ colorspace_code['ycbcr'], colorspace_code['lab'] ]
    
    color_alphas = [[0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 1.0, 1.0]]
    c_alpha = []
    for a in color_alphas:
        norm_weight = normalize_color_weights('ycbcr', a[0], a[1], a[2])
        
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
        
        norm_weight = normalize_color_weights('lab', a[0], a[1], a[2])
          
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
    c_alpha.append( [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ])    
    
    combinations = list(product(c_spatial_radius, c_alpha, c_colorspace, c_volume_threshold, c_bins_per_band, c_beta))
    params += [ { 'spatial_radius' : tuple[0], 'alpha' : tuple[1], 
                'colorspace' : tuple[2], 'volume_threshold' : tuple[3],
                'bins_per_band' : tuple[4], 'beta' : tuple[5]  } for tuple in combinations]
    #RGB
    c_colorspace = [ colorspace_code['rgb' ]]
    
    c_alpha = []
    norm_weight = normalize_color_weights('rgb',1.0, 1.0,1.0)
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
    c_alpha.append( [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ])    
     
    combinations = list(product(c_spatial_radius, c_alpha, c_colorspace, c_volume_threshold, c_bins_per_band, c_beta))
    params += [ { 'spatial_radius' : tuple[0], 'alpha' : tuple[1], 
                'colorspace' : tuple[2], 'volume_threshold' : tuple[3],
                'bins_per_band' : tuple[4], 'beta' : tuple[5]  } for tuple in combinations]
    
    print(len(params))
    
    f = partial(gc_super_validation_func, exp_params = exp_params)
    
    pool = Pool()
    results = numpy.array(pool.map(f, params))
    
    write_results(output_file, params, results)
    
def slic_ift_super_validation_func(params, exp_params):
    geodesic_robot= ef.GeodesicRobotExperiment(exp_params, {})
    
    super_seg = ef.SLICSuperpixelSegmentation(params)
    
    stats = geodesic_robot.run(super_seg)
    fscores = [ s['fscore'] for s in stats]
    
    return metrics.auc(numpy.linspace(0.,1.,len(fscores)),fscores)    
    
def slic_ift_super_validation_experiment(exp_params, output_file):
    """alpha - list containing 7 elements weighting: mean band1, mean band2, mean band3, relative area, color histogram intersection, lbp histogram, rectangularity"""
    colorspace_code = { 'ycbcr' : 0, 'rgb' : 1, 'lab' : 4}    
     
    params = []
    
    c_compactness = [ 10 ]
    c_nregions = [ 300, 600]
    c_bins_per_band = [ 4, 8]
    
    c_colorspace = [ colorspace_code['ycbcr'], colorspace_code['lab'] ]
     
    color_alphas = [[0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 1.0, 1.0]]
    c_alpha = []
    for a in color_alphas:
        norm_weight = normalize_color_weights('ycbcr', a[0], a[1], a[2])
          
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
        
        norm_weight = normalize_color_weights('lab', a[0], a[1], a[2])
          
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
    c_alpha.append( [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ])   
      
    combinations = list(product(c_compactness, c_alpha, c_colorspace, c_nregions, c_bins_per_band))
    params += [ { 'compactness' : tuple[0], 'alpha' : tuple[1], 
                'colorspace' : tuple[2], 'nregions' : tuple[3],
                'bins_per_band' : tuple[4]  } for tuple in combinations]
    #RGB
    c_colorspace = [ colorspace_code['rgb' ]]
     
    c_alpha = []
    norm_weight = normalize_color_weights('rgb',1.0, 1.0,1.0)
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
    c_alpha.append( [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ])    
      
    combinations = list(product(c_compactness, c_alpha, c_colorspace, c_nregions, c_bins_per_band))
    params += [ { 'compactness' : tuple[0], 'alpha' : tuple[1], 
                'colorspace' : tuple[2], 'nregions' : tuple[3],
                'bins_per_band' : tuple[4]  } for tuple in combinations]
      
    print(len(params))
      
    f = partial(slic_ift_super_validation_func, exp_params = exp_params)
      
    pool = Pool()
    results = numpy.array(pool.map(f, params))
      
    write_results(output_file, params, results)

def slic_gc_super_validation_func(params, exp_params):
    geodesic_robot= ef.GeodesicRobotExperiment(exp_params, {})
    
    gc_super_seg = ef.SLICGCSuperSegmentation(params)
    
    stats = geodesic_robot.run(gc_super_seg)
    fscores = [ s['fscore'] for s in stats]
    
    return metrics.auc(numpy.linspace(0.,1.,len(fscores)),fscores)    
    
def slic_gc_super_validation_experiment(exp_params, output_file):
    colorspace_code = { 'ycbcr' : 0, 'rgb' : 1, 'lab' : 4}    
    
    params = []
    
    c_compactness = [ 10 ]
    c_nregions = [ 300, 600]
    c_bins_per_band = [ 4, 8]
    c_beta = [1,10,20,30]
    
    #YCbCr and LAB
    c_colorspace = [ colorspace_code['ycbcr'], colorspace_code['lab'] ]
    
    color_alphas = [[0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 1.0, 1.0]]
    c_alpha = []
    for a in color_alphas:
        norm_weight = normalize_color_weights('ycbcr', a[0], a[1], a[2])
        
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
        
        norm_weight = normalize_color_weights('lab', a[0], a[1], a[2])
          
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
        c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
    c_alpha.append( [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ])    
    
    combinations = list(product(c_compactness, c_alpha, c_colorspace, c_nregions, c_bins_per_band, c_beta))
    params += [ { 'compactness' : tuple[0], 'alpha' : tuple[1], 
                'colorspace' : tuple[2], 'nregions' : tuple[3],
                'bins_per_band' : tuple[4], 'beta' : tuple[5]  } for tuple in combinations]
    #RGB
    c_colorspace = [ colorspace_code['rgb' ]]
    
    c_alpha = []
    norm_weight = normalize_color_weights('rgb',1.0, 1.0,1.0)
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 0.0, 0.0 ])
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 0.0, 1.0, 0.0 ]) 
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 0.0, 0.0 ]) 
    c_alpha.append( [ norm_weight[0], norm_weight[1], norm_weight[2], 0.0, 1.0, 1.0, 0.0 ])
    c_alpha.append( [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ])    
     
    combinations = list(product(c_compactness, c_alpha, c_colorspace, c_nregions, c_bins_per_band, c_beta))
    params += [ { 'compactness' : tuple[0], 'alpha' : tuple[1], 
                'colorspace' : tuple[2], 'nregions' : tuple[3],
                'bins_per_band' : tuple[4], 'beta' : tuple[5]  } for tuple in combinations]
    
    print(len(params))
    
    f = partial(slic_gc_super_validation_func, exp_params = exp_params)
    
    pool = Pool()
    results = numpy.array(pool.map(f, params))
    
    write_results(output_file, params, results)

def main():
    parser = argparse.ArgumentParser('Segmentation validation experiments')
    
    parser.add_argument('dataset_path', type = str, help = 'Path to the segmentation dataset')
    parser.add_argument('output_path', type = str, help = 'Path to write the results')
    
    seg_methods = { 'gc_pixel' : gc_pixel_validation_experiment, 
                   'gc_super': gc_super_validation_experiment,
                   'ift_pixel' : ift_pixel_validation_experiment,
                   'ift_super' : ift_super_validation_experiment,
                   'slic_ift_super' : slic_ift_super_validation_experiment,
                   'slic_gc_super': slic_gc_super_validation_experiment
                   }
    
    args = parser.parse_args()
    
    exp_params = {   
       'dataset_path'         : args.dataset_path,
       'number_iterations'    : 8,
       'seeds_per_iteration'  : 8,
       'max_marker'           : 8,
       'min_marker'           : 1,
       'safe_distance'        : 2,
       }
    
    #FIXME:
    for method in ['slic_gc_super', 'slic_ift_super']:
    #for method in seg_methods.keys():
        output_file = os.path.join(args.output_path, method + '_val.txt') 
        seg_methods[method](exp_params, output_file)

if __name__ == "__main__":
    main()
