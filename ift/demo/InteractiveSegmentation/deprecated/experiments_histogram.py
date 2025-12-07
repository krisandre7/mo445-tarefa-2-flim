#!/usr/bin/python3.3
#experiments.py: Robot user experiments
import os
from pyift import *

class Experiment:
    def __init__(self, images_path, spatial_radius, vol_thr, max_iter, seeds_iter, max_mark, min_mark, dist_bord,vol_robot,vol_histo,nbins,border_dist):
        self.current_images = [f for f in os.listdir(images_path) if f.endswith(".ppm")]
        self.current_folder = images_path
   
        self.spatial_radius = spatial_radius
        self.volume_threshold = vol_thr
        
        self.MAX_ITERATIONS = max_iter
        self.SEEDS_PER_ITERATION = seeds_iter
        
        self.MAX_MARKER_SIZE = max_mark
        self.MIN_MARKER_SIZE = min_mark
        self.DIST_FROM_BORDER = dist_bord
        
        self.vol_robot = vol_robot
        self.vol_histo = vol_histo
        self.nbins = nbins
        
        self.border_dist = border_dist
 
    def create_superpixel_cl_image(self, image, seed, spatial_radius, volume_threshold ):
        #Segmentation
        adj_relation = iftCircular(spatial_radius);
        basins = iftImageBasins(image,adj_relation)
        marker = iftVolumeClose(basins,volume_threshold)
        
        #Label Image
        label_image = iftWaterGray(basins,marker,adj_relation)
        
        dataset = iftSupervoxelsToDataSet(image, label_image)
        if dataset.nfeats == 3:
            dataset.set_alpha([0.2,1.0,1.0])
                
        adj_relation = iftCircular(1)
        region_graph = iftRegionGraphFromLabelImage(label_image,dataset,adj_relation)
        iftSuperpixelClassification(region_graph,label_image, seed);         
                
        #Creating classification mask
        classification_image = iftCreateImage(image.xsize,image.ysize,image.zsize)
        for p in range(0,classification_image.n):
            #region label
            pixel_label = label_image[p][0] - 1
            #classification label
            color = dataset.get_sample_label(pixel_label)
            classification_image[p] = (color, 0, 0)
        
        iftWriteSeedsOnImage(classification_image,seed)
        
        return classification_image
        
    def create_histogram_cl_image(self, image, seed, spatial_radius, volume_threshold,nbins):
        #Segmentation
        adj_relation = iftCircular(spatial_radius);
        basins = iftImageBasins(image,adj_relation)
        marker = iftVolumeClose(basins,volume_threshold)
        
        #Label Image
        label_image = iftWaterGray(basins,marker,adj_relation)
        
        dataset = iftSupervoxelsToHistogramDataSet(image, label_image,nbins)
                        
        adj_relation = iftCircular(1)
        region_graph = iftRegionGraphFromLabelImage(label_image,dataset,adj_relation)
        iftSuperpixelClassification(region_graph,label_image, seed);         
                
        #Creating classification mask
        classification_image = iftCreateImage(image.xsize,image.ysize,image.zsize)
        for p in range(0,classification_image.n):
            #region label
            pixel_label = label_image[p][0] - 1
            #classification label
            color = dataset.get_sample_label(pixel_label)
            classification_image[p] = (color, 0, 0)
        
        iftWriteSeedsOnImage(classification_image,seed)
        
        return classification_image
    
    def _make_stats_string(self, method, stats):
        return method + "(precision: {0:.3f}, recall: {1:.3f}, fscore: {2:.3f}, accuracy: {3:.3f}) \n".format(stats[0],stats[1],stats[2],stats[3])

    def _get_stats_string_from_errors(self, method, errors):
        precision = iftPrecisionGivenErrors(errors)
        recall = iftRecallGivenErrors(errors)
        fscore = iftFScoreGivenErrors(errors)
        accuracy = iftAccuracyGivenErrors(errors) 
        
        return self._make_stats_string(method, [precision,recall,fscore,accuracy])
    
    #precision, recall, fscore, accuracy
    def _acc_errors(self, stats, errors):
        precision = iftPrecisionGivenErrors(errors)
        recall = iftRecallGivenErrors(errors)
        fscore = iftFScoreGivenErrors(errors)
        accuracy = iftAccuracyGivenErrors(errors) 
        
        return [stats[0] + precision, stats[1] + recall, stats[2] + fscore, stats[3] + accuracy]

    
    def run_robot(self, ROBOT, result_path):
       #File where we'll write the statistics
        file = open(result_path,"w")
        file.write("#" + ROBOT + " Robot:\n")
        file.write("#Iterations: {0}, Seeds per iteration: {1}, Max Marker Size: {2}, Min Marker Size: {3},  Distance from Border: {4}".format(self.MAX_ITERATIONS,self.SEEDS_PER_ITERATION,self.MAX_MARKER_SIZE,self.MIN_MARKER_SIZE,self.DIST_FROM_BORDER))
        file.write("### Statistics by image:\n")
        
        #Each element will hold the cumulative statistics for each iteration
        acc_stats = []
        for i in range(self.MAX_ITERATIONS):
            acc_stats.append({'MEAN': [0.0,0.0,0.0,0.0], 'HISTO' : [0.0,0.0,0.0,0.0]})
        
        average_seeds_super = 0
        average_seeds_pixel = 0
        
        n = 0
        for img in self.current_images:
            nseeds_sup = 0
            nseeds_pix = 0
            
            n += 1
            
            image = iftReadImageP6(os.path.join(self.current_folder,img))
            gt_image = iftReadImageP5(os.path.join(self.current_folder,"gt/" + img[:-3] + "pgm"))
        
            #Creates the initial classification images
            super_cl_image = None
            pixel_cl_image = None   
            
            super_seeds_bmap = iftCreateBMap(image.n)
            pixel_seeds_bmap = iftCreateBMap(image.n)
            
            #Creading the seeds for superpixel and pixel robots
            if ROBOT == "SUPER":
                adj_relation = iftCircular(self.spatial_radius)
                basins = iftImageBasins(image,adj_relation)
                marker = iftVolumeClose(basins,self.vol_robot)
                label_image = iftWaterGray(basins,marker,adj_relation)
                dataset = iftSupervoxelsToDataSet(image, label_image)
                if dataset.nfeats == 3:
                    dataset.set_alpha([0.2,1.0,1.0])
                
                available_seeds = iftBorderMarkersForSuperpixelSegmentation(label_image,gt_image, dataset)
            elif ROBOT == "PIXEL":
                adj_relation = iftCircular(self.spatial_radius)
                basins = iftImageBasins(image,adj_relation)
                available_seeds = iftBorderMarkersForPixelSegmentation(basins,gt_image,self.border_dist)
            
            seed_sup_image = iftCreateImage(image.xsize,image.ysize, image.zsize)
            seed_pix_image = iftCreateImage(image.xsize,image.ysize, image.zsize)
            
            #Iteratively selects new seeds to improve classification 
            for i in range(self.MAX_ITERATIONS):
                if ROBOT == "SUPER" or ROBOT == "PIXEL":
                    nseeds_sup += iftMarkersFromMisclassifiedSeeds(seed_sup_image,available_seeds,super_seeds_bmap,self.SEEDS_PER_ITERATION,gt_image,super_cl_image,self.DIST_FROM_BORDER,self.MAX_MARKER_SIZE,self.MIN_MARKER_SIZE) 
                    nseeds_pix += iftMarkersFromMisclassifiedSeeds(seed_pix_image,available_seeds,pixel_seeds_bmap, self.SEEDS_PER_ITERATION,gt_image,pixel_cl_image,self.DIST_FROM_BORDER,self.MAX_MARKER_SIZE,self.MIN_MARKER_SIZE) 
                elif ROBOT == "GEODESIC":
                    available_seeds = iftGeodesicMarkersForSegmentation(gt_image, super_cl_image)
                    nseeds_sup += iftMarkersFromMisclassifiedSeeds(seed_sup_image,available_seeds,super_seeds_bmap,self.SEEDS_PER_ITERATION,gt_image,super_cl_image,self.DIST_FROM_BORDER,self.MAX_MARKER_SIZE,self.MIN_MARKER_SIZE) 
                    
                    available_seeds = iftGeodesicMarkersForSegmentation(gt_image, pixel_cl_image)
                    nseeds_pix += iftMarkersFromMisclassifiedSeeds(seed_pix_image,available_seeds,pixel_seeds_bmap,self.SEEDS_PER_ITERATION,gt_image,pixel_cl_image,self.DIST_FROM_BORDER,self.MAX_MARKER_SIZE,self.MIN_MARKER_SIZE) 

                seed_sup = iftLabeledSetFromSeedImage(seed_sup_image)
                seed_pix = iftLabeledSetFromSeedImage(seed_pix_image)
                
                super_cl_image = self.create_superpixel_cl_image(image, seed_sup, self.spatial_radius, self.volume_threshold) 
                pixel_cl_image = self.create_histogram_cl_image(image, seed_pix, self.spatial_radius, self.vol_histo, self.nbins)

                super_errors = iftClassifyBinarySegmentationErrors(gt_image,super_cl_image)
                pixel_errors = iftClassifyBinarySegmentationErrors(gt_image,pixel_cl_image)
                
                acc_stats[i]["MEAN"] = self._acc_errors(acc_stats[i]["MEAN"], super_errors)
                acc_stats[i]["HISTO"] = self._acc_errors(acc_stats[i]["HISTO"], pixel_errors)
                
                stats_string = img + ", iteration {0}:\n".format(i+1)
                stats_string +=  self._get_stats_string_from_errors("MEAN", super_errors)
                stats_string +=  self._get_stats_string_from_errors("HISTO", pixel_errors)
                stats_string += "\n"
               
                #file.write(stats_string)
                print(stats_string)
                
                #Writes some set of seeds to disk
#                 iftWriteSeeds2D(os.path.join(self.current_folder,img)[:-3] + "markers",seed_sup,image)
                    
                #Write partial markers to disk (debugging)
#                if not os.path.exists("tmp/"):
#                    os.makedirs("tmp/")
#                    
#                colors = [ (225,1,148),(76,85,255) ]
#                
#                image_sup_markers = iftCopyImage(image)
#                image_pix_markers = iftCopyImage(image)
#                
#                for p in range(image.n):
#                    if seed_sup_image[p] != (0,0,0):
#                        image_sup_markers[p] = colors[seed_sup_image[p][0] - 1]
#                    if seed_pix_image[p] != (0,0,0):
#                        image_pix_markers[p] = colors[seed_pix_image[p][0] - 1]
#                    
#                image_sup_seg = iftCopyImage(image_sup_markers)
#                image_pix_seg = iftCopyImage(image_pix_markers)
#                    
#                for p in range(image.n):
#                    if super_cl_image[p] == (1,0,0):
#                        image_sup_seg[p] = (image_sup_seg[p][0],colors[0][1],colors[0][2])
#                    else:
#                        image_sup_seg[p] = (image_sup_seg[p][0],colors[1][1],colors[1][2])
#                    if pixel_cl_image[p] == (1,0,0):
#                        image_pix_seg[p] = (image_pix_seg[p][0],colors[0][1],colors[0][2])
#                    else:
#                        image_pix_seg[p] = (image_pix_seg[p][0],colors[1][1],colors[1][2])
#                        
#                iftWriteImageP6(image_sup_markers,"tmp/" + img[:-4] + "_m_mean_" + str(i+1) + "_R_"+ROBOT + ".ppm")
#                iftWriteImageP6(image_pix_markers,"tmp/" + img[:-4] + "_m_hist_" + str(i+1) + "_R_"+ROBOT + ".ppm")
#                iftWriteImageP6(image_sup_seg,"tmp/" + img[:-4] + "_s_mean_" + str(i+1) + "_R_"+ROBOT + ".ppm")
#                iftWriteImageP6(image_pix_seg,"tmp/" + img[:-4] + "_s_histo_" + str(i+1) + "_R_"+ROBOT + ".ppm")
                
            average_seeds_super += nseeds_sup
            average_seeds_pixel += nseeds_pix
            
        average_seeds_super = average_seeds_super/n
        average_seeds_pixel = average_seeds_pixel/n
            
        stats_string = '### Cumulative statistics: \n'
        stats_string += "#Iter    Mean    Histo\n"
        for i in range(self.MAX_ITERATIONS):
            acc_stats[i]["MEAN"] = [j/n for j in acc_stats[i]["MEAN"]]
            acc_stats[i]["HISTO"] = [j/n for j in acc_stats[i]["HISTO"]]
                
#            stats_string += "Iteration {}:\n".format(i+1)
#            stats_string += self._make_stats_string("SUPER", acc_stats[i]["SUPER"])
#            stats_string += self._make_stats_string("PIXEL", acc_stats[i]["PIXEL"])
#            stats_string += "\n"

            stats_string += str(i+1) + "    " + "{0:.3f}".format(acc_stats[i]["MEAN"][2]) + "    " + "{0:.3f}".format(acc_stats[i]["HISTO"][2]) + "\n"
            
        stats_string += "#MEAN(Average seeds per image: {0})\n".format(average_seeds_super)
        stats_string += "#HISTO(Average seeds per image: {0})\n".format(average_seeds_pixel)
            
        print(stats_string)    
        file.write(stats_string)
        
        file.close()           
        print("Finished.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def save_graphics(p,filename):
    #loads 
    files = [ "grabcut_statistics_sup.txt", "grabcut_statistics_pix.txt", "grabcut_statistics_geo.txt",
             "geostar_statistics_sup.txt", "geostar_statistics_pix.txt", "geostar_statistics_geo.txt",
             "weizmann_statistics_sup.txt", "weizmann_statistics_pix.txt", "weizmann_statistics_geo.txt"]
    
    plt.figure(figsize=(15,15))
        
    i = 0
    for file in files:
        iters = []
        supers = []
        pixels = []
        
        f = open(file,"r")
        for line in f:
            if line[0] != "#":
                l = line.split()
                iters.append(int(l[0]))
                supers.append(float(l[1]))
                pixels.append(float(l[2]))

        axis = plt.subplot(3,3,i+1)
        axis.set_title(file[:-4])
        
        #super
        axis.plot(iters,supers, label = "Mean color", marker="o")
        #pixel
        axis.plot(iters,pixels, label = "Histogram",marker = "o")
        
        plt.ylim((0.5,1.0))
        plt.xticks(range(1,max(iters) + 1))
        
        axis.legend(loc=4)
        axis.grid()
        
        i += 1
        
    plt.savefig(filename)
    
def main():
    if not os.path.exists("stats_histo/"):
        os.makedirs("stats_histo/")
        
    exp_param = [ 
				  #Baseline
                  #(0 SPATIAL_RADIUS,1 VOLUME_THRESHOLD,2 MAX_ITERATIONS,3 SEED_PER_ITERATION,4 MAX_MARKER_SIZE,5 MIN_MARKER_SIZE, 6 MIN_DIST_FROM_BORDER, 7 VOLUME_SUPER_ROBOT,8 VOLUME_HISTO, 9 NBINS, 10 BORDER_DISTANCE)
				  (1.5, 100, 2, 20, 5, 1, 2,15000, 100,64,15)
				  ]
    
    for p in exp_param:
        filename = "stats_histo/{0}it{1}seed{2}mmax{3}mmin{4}mdist{5}vrob{6}spr{7}vth{8}vhisto{9}nbins{10}bdist.png".format(p[2],p[3],p[4],p[5],p[6],p[7],p[0],p[1],p[8],p[9],p[10])
        if not os.path.exists(filename):
            exp = Experiment("grabcut/",*p)
            exp.run_robot("SUPER","grabcut_statistics_sup.txt")
            exp.run_robot("PIXEL", "grabcut_statistics_pix.txt")
            exp.run_robot("GEODESIC","grabcut_statistics_geo.txt")
             
            exp = Experiment("geostar/",*p)
            exp.run_robot("SUPER","geostar_statistics_sup.txt")
            exp.run_robot("PIXEL", "geostar_statistics_pix.txt")
            exp.run_robot("GEODESIC","geostar_statistics_geo.txt")
             
            exp = Experiment("weizmann/",*p)
            exp.run_robot("SUPER","weizmann_statistics_sup.txt")
            exp.run_robot("PIXEL", "weizmann_statistics_pix.txt")
            exp.run_robot("GEODESIC","weizmann_statistics_geo.txt")
            
            save_graphics(p,filename)

if __name__ == '__main__':
    main()
