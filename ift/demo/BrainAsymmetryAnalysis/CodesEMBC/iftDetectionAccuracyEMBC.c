#include "ift.h"


int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("%s <detection_mask_dir> <hemisphere_mask_dir> <lesion_gt_dir>", "main", argv[0]);
    
    const char *det_mask_dir = argv[1];
    const char *hem_mask_dir = argv[2];
    const char *lesion_gt_dir = argv[3];
    
    printf("- Detection Mask Dir: %s\n", det_mask_dir);
    printf("- Hemisphere Mask Dir: %s\n", hem_mask_dir);
    printf("- Lesion GT Dir: %s\n", lesion_gt_dir);
    puts("------------\n");
    
    iftFileSet *det_set = iftLoadFileSetFromDirOrCSV(det_mask_dir, 0, true);
    
    iftFSet *recalls_set = NULL; // true positive rates
    int n_detected_lesions = 0;
    int n_total_lesions = 0;
    
    
    for (int i = 0; i < det_set->n; i++) {
        char *filename = iftFilename(det_set->files[i]->path, NULL);
        char *hem_mask_path = iftJoinPathnames(2, hem_mask_dir, filename);
        char *lesion_gt_path = iftJoinPathnames(2, lesion_gt_dir, filename);
        
        printf("[%d]\n", i);
        printf("Detected Mask: %s\n", det_set->files[i]->path);
        printf("Hemisphere Mask: %s\n", hem_mask_path);
        printf("Lesion GT: %s\n", lesion_gt_path);
        
        iftImage *det_img = iftReadImageByExt(det_set->files[i]->path);
        iftImage *hem_mask = iftReadImageByExt(hem_mask_path);
        iftImage *lesion_gt = iftReadImageByExt(lesion_gt_path);
        iftImage *lesion_gt_in_hems = iftMask(lesion_gt, hem_mask);
        
        if (iftMaximumValue(lesion_gt_in_hems)) {
            iftImage *lesion_gt_final = iftRelabelImage(lesion_gt_in_hems);
            
            int n_lesions = iftMaximumValue(lesion_gt_final);
            n_total_lesions += n_lesions;
    
            iftFloatArray *TP = iftCreateFloatArray(n_lesions + 1);
            iftIntArray *n_voxels = iftCreateIntArray(n_lesions + 1);
    
            for (int p = 0; p < lesion_gt_final->n; p++) {
                int label = lesion_gt_final->val[p];
                n_voxels->val[label]++;
        
                if ((label) && (det_img->val[p]))
                    TP->val[label]++;
            }
    
            for (int label = 1; label <= n_lesions; label++) {
                float recall = TP->val[label] / n_voxels->val[label]; // true positive rate
                iftInsertFSet(&recalls_set, recall);
        
                if (recall >= 0.15)
                    n_detected_lesions++;
        
                printf("----------\nLabel: %d\n", label);
                printf("n_voxels: %d, recall: %f\n", n_voxels->val[label], recall);
            }
            puts("\n");
    
            iftDestroyImage(&lesion_gt_final);
            iftDestroyFloatArray(&TP);
            iftDestroyIntArray(&n_voxels);
        }
        else
            printf("***** Lesion Mask without any lesion with at least a part inside the hemisphere masks *****\n\n");
    
        iftFree(filename);
        iftFree(hem_mask_path);
        iftFree(lesion_gt_path);
        iftDestroyImage(&det_img);
        iftDestroyImage(&hem_mask);
        iftDestroyImage(&lesion_gt);
        iftDestroyImage(&lesion_gt_in_hems);
    }
    
    
    iftFloatArray *recalls_arr = iftCreateFloatArray(n_total_lesions);
    int i = 0;
    while (recalls_set) {
        float recall = iftRemoveFSet(&recalls_set);
        recalls_arr->val[i] = recall;
        i++;
    }
    
    float mean_recall = iftMean(recalls_arr->val, recalls_arr->n);
    float stdev_recall = iftStd(recalls_arr->val, recalls_arr->n);
    float perc_detected_lesions = n_detected_lesions / ((float) n_total_lesions);
    
    printf("\n\n-> Mean Recall: %f +- %f\nPerc. Detected Lesions: %d/%d = %f\n",
           mean_recall, stdev_recall, n_detected_lesions, n_total_lesions, perc_detected_lesions);
    
    iftDestroyFileSet(&det_set);
    iftDestroyFloatArray(&recalls_arr);
    
    return 0;
}
