import pascal
import detection
import numpy

p = pascal.Pascal('/workspace/paulorauber/projects/ift/data/pascal')

def evaluate_recall_regions(positive_class, images, regions, plot = True):
    print('Images in class {0}: {1}.'.format(positive_class, len(images)))
    
    frames = p.get_positive_frames(images, positive_class)
    
    print('Regions/Image: {0}.'.format(len(regions)/len(images)))
    recall = p.evaluate_recall(frames, regions, plot = plot)
    print('Recall: {0}.'.format(recall))
    
    return recall

def evaluate_random_window(positive_class, regions_per_image, plot = True):
    images = p.get_positive_images(p.validation_list, positive_class)
    return evaluate_recall_regions(positive_class, images, p.get_regions_random(images, regions_per_image), plot = plot)
    
def evaluate_kde_window(positive_class, regions_per_image, plot = True):
    images = p.get_positive_images(p.validation_list, positive_class)
    return evaluate_recall_regions(positive_class, images, p.get_regions_kde(images, positive_class, regions_per_image, verbose = False), plot = plot)
    
def evaluate_superpixel_window(positive_class, regions_per_image, plot = True, vol_threshold = 150, filter_area = 100, nbins = 25, alpha = [0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 0.0, 0.0, 0.0]):
    images = p.get_positive_images(p.validation_list, positive_class)
    return evaluate_recall_regions(positive_class, images, p.get_regions_superpixels(images, vol_threshold, regions_per_image, nbins,  alpha, verbose = False), plot = plot)
    
def evaluate_all(regions_per_image):
    classes = p.classes
    
    scores = {}
    for c in classes:
        scores[c] = (evaluate_random_window(c, regions_per_image, False), evaluate_kde_window(c, regions_per_image, False), evaluate_superpixel_window(c, regions_per_image, False))
        print("Scores for class {0} (r/k/s): {1}".format(c, scores[c]))
        
    return scores
