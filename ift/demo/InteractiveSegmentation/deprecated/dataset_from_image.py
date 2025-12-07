#!/usr/bin/python3.3
from pyift import *
import sys

def main(spatial_radius = 1.5, volume_threshold = 150):
    if len(sys.argv) != 4:
        raise Exception("Usage: dataset_from_image [input_image] [input_gt] [output_dataset]")
    
    image = iftReadImageP6(sys.argv[1])
    gt_image = iftReadImageP5(sys.argv[2])
    
    adj_relation = iftCircular(spatial_radius);
    basins = iftImageBasins(image,adj_relation)
    marker = iftVolumeClose(basins,volume_threshold)
    
    #Label Image
    label_image = iftWaterGray(basins,marker,adj_relation)
    
    dataset = iftSupervoxelsToDataSet(image, label_image)
    if dataset.nfeats == 3:
        dataset.set_alpha([0.2,1.0,1.0])
        
    for i in range(label_image.n):
        if gt_image[i][0] != 0:
            dataset.set_class(label_image[i][0] - 1,2)
        else:
            dataset.set_class(label_image[i][0] - 1,1)
        
    iftWriteOPFDataSet(dataset, sys.argv[3])

if __name__ == "__main__":
    main()