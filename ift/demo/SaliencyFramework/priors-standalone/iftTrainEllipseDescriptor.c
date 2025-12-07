#include "ift.h"

#include "iftGraphSaliency.c"

iftImage **readImageSet(iftFileSet *fs);

int main(int argc, const char *argv[]) {

    if (argc != 4) {
        iftError("Usage: iftTrainEllipseDescriptor <orig_path> <gt_path> <parameter file>", "main");
    }

    iftImage *mask, *seeds, *label_img;
    iftIGraph *igraph, *out_igraph;
    float alpha = 0.8, beta = 12, gamma = 5.0, iter = 1, obj_perc = 0.01;
    int number_superpixels = 100;
    iftFileSet *fs_orig = iftLoadFileSetFromDirOrCSV(argv[1], 1, 1);
    iftFileSet *fs_gt = iftLoadFileSetFromDirOrCSV(argv[2], 1, 1);


    iftImage **orig = readImageSet(fs_orig);
    iftImage **gt = readImageSet(fs_gt);

    iftWriteImageByExt(gt[0], "test.png");
    int nimages = fs_orig->n;
    int nimages_gt = fs_gt->n;

    float mean_minor = 0, mean_anisotropy = 0, mean_major = 0;

    int superpixel_counter = 0;
    printf("Computing the mean\n");
    for(int i = 0; i < nimages; i++){
        mask = iftSelectImageDomain(orig[i]->xsize, orig[i]->ysize, orig[i]->zsize);
        seeds = iftSamplingByOSMOX(gt[i], mask, number_superpixels, obj_perc);
        igraph = iftInitOISFIGraph(orig[i], mask, gt[i]);
        out_igraph = iftIGraphOISF(igraph, seeds, alpha, beta, gamma, iter);

        label_img = iftIGraphLabel(out_igraph);
        iftVoxelArray *centroids = iftGeometricCentersFromLabelImage(label_img);

        iftTensorScale *tensor_scale = iftSuperpixelToTensorScale(label_img, 8);

        for(int s = 0; s < centroids->n-1; s++){
            mean_minor += tensor_scale->minor_axis->val[s];
            mean_major += tensor_scale->major_axis->val[s];
            mean_anisotropy += tensor_scale->anisotropy->val[s];
            superpixel_counter++;
        }
        iftDestroyVoxelArray(&centroids);
        iftDestroyImage(&mask);
        iftDestroyImage(&seeds);
        iftDestroyIGraph(&igraph);
        iftDestroyIGraph(&out_igraph);
    }

    mean_minor/=superpixel_counter;
    mean_major/=superpixel_counter;
    mean_anisotropy/=superpixel_counter;

    float std_minor = 0, std_anisotropy = 0, std_major = 0;
    superpixel_counter = 0;
    printf("Computing the std\n");
    for(int i = 0; i < nimages; i++){
        mask = iftSelectImageDomain(orig[i]->xsize, orig[i]->ysize, orig[i]->zsize);
        seeds = iftSamplingByOSMOX(gt[i], mask, number_superpixels, obj_perc);
        igraph = iftInitOISFIGraph(orig[i], mask, gt[i]);
        out_igraph = iftIGraphOISF(igraph, seeds, alpha, beta, gamma, iter);

        label_img = iftIGraphLabel(out_igraph);
        iftVoxelArray *centroids = iftGeometricCentersFromLabelImage(label_img);

        iftTensorScale *tensor_scale = iftSuperpixelToTensorScale(label_img, 8);

        for(int s = 0; s < centroids->n-1; s++){
            std_minor += pow((tensor_scale->minor_axis->val[s] - mean_minor), 2);
            std_major += pow((tensor_scale->major_axis->val[s] - mean_major), 2);
            std_anisotropy += pow((tensor_scale->anisotropy->val[s] - mean_anisotropy), 2);
            superpixel_counter++;
        }
        iftDestroyVoxelArray(&centroids);
        iftDestroyImage(&mask);
        iftDestroyImage(&seeds);
        iftDestroyIGraph(&igraph);
        iftDestroyIGraph(&out_igraph);

    }

    float variance_minor = std_minor/superpixel_counter;
    float variance_major = std_major/superpixel_counter;
    float variance_anisotropy = std_anisotropy/superpixel_counter;
    std_minor = sqrtf(variance_minor);
    std_major = sqrtf(variance_major);
    std_anisotropy = sqrtf(variance_anisotropy);

    char filename[200];

    strcpy(filename, argv[3]);

    FILE *fp = fopen(filename,"w");

    fprintf(fp,"%f %f\n",mean_minor, mean_anisotropy);
    fprintf(fp,"%f %f\n",mean_major, mean_anisotropy);
    fprintf(fp,"%f %f",std_minor, std_anisotropy);

    fclose(fp);

    iftDestroyFileSet(&fs_gt);
    iftDestroyFileSet(&fs_orig);
    iftDestroyImage(&label_img);

    for(int i = 0; i < nimages; i++){
        iftDestroyImage(&orig[i]);
        iftDestroyImage(&gt[i]);
    }

    free(orig);
    free(gt);
}

iftImage **readImageSet(iftFileSet *fs){
    int first = 0;
    int last  = fs->n - 1;

    iftImage **orig = (iftImage **) calloc(fs->n, sizeof(iftImage *));

    for (int i = first; i <= last; i++) {
        char *basename = iftFilename(fs->files[i]->path, iftFileExt(fs->files[i]->path));
        printf("Reading file %s: %d of %ld files\r", basename, i + 1, fs->n);
        fflush(stdout);
        orig[i] = iftReadImageByExt(fs->files[i]->path); // it seems better than YCbCrNorm_CSPACE and LAB_CSPACE
    }
    return(orig);
}