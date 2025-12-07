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
        printf("Image %d\n", i);
        mask = iftSelectImageDomain(orig[i]->xsize, orig[i]->ysize, orig[i]->zsize);
        seeds = iftSamplingByOSMOX(gt[i], mask, number_superpixels, obj_perc);
        igraph = iftInitOISFIGraph(orig[i], mask, gt[i]);
        out_igraph = iftIGraphOISF(igraph, seeds, alpha, beta, gamma, iter);

        label_img = iftIGraphLabel(out_igraph);
        iftVoxelArray *centroids = iftGeometricCentersFromLabelImage(label_img);

        iftTensorScale *tensor_scale = iftSuperpixelToTensorScale(label_img, 10);
        iftImage *ellipsesImage = iftCreateImageFromImage(label_img);
        iftIntArray *ellipseError = iftCreateIntArray(centroids->n-1);
        iftFloatArray *ellipseMatch = iftCreateFloatArray(centroids->n-1);
        for(int p = 0; p < label_img->n; p++) {
                    iftVoxel u = iftGetVoxelCoord(gt[i], p);
                    int s = label_img->val[p]-1;
                    float ellipse_eq = sqrtf(powf((u.x - tensor_scale->pos_focus->val[s].x), 2) + (powf((u.y - tensor_scale->pos_focus->val[s].y), 2))) + sqrtf(powf((u.x - tensor_scale->neg_focus->val[s].x), 2) + (powf((u.y - tensor_scale->neg_focus->val[s].y), 2)));
                    if (ellipse_eq > 2*tensor_scale->major_axis->val[s]) {
                        ellipseError->val[s] += 1;
                        ellipsesImage->val[p] = 0;
                    }else{
                        ellipseMatch->val[s]+=1;
                        ellipsesImage->val[p] = 255;
                    }
        }

        iftIntArray *region_sizes;

        region_sizes = iftCountLabelSpels(label_img);

        for(int s = 0; s < ellipseMatch->n; s++){
            ellipseMatch->val[s] = exp( (ellipseMatch->val[s] / region_sizes->val[s+1]) / 0.1);
            if(region_sizes->val[s+1] < 300 || region_sizes->val[s+1] > 6000) //Random values. Have to change that
                ellipseMatch->val[s] /= 10;
        }

        float max=0, min=IFT_INFINITY_INT;
        for(int p=0; p < ellipseMatch->n; p++){
            if(ellipseMatch->val[p] > max)
                max = ellipseMatch->val[p];
            if(ellipseMatch->val[p] < min)
                min = ellipseMatch->val[p];
        }

        printf("Max %f Min %f", max, min);

        for(int p = 0; p < ellipseMatch->n; p++)
            ellipseMatch->val[p] = 1 * (ellipseMatch->val[p] - min) / (max - min);

        iftImage *ellipseMatchImage = iftCreateImageFromImage(label_img);
        for(int p = 0; p < label_img->n; p++)
            ellipseMatchImage->val[p] = (int)(255 * ellipseMatch->val[label_img->val[p]-1]);
        char filename[200];

        sprintf(filename, "ellipse%d.png", i);
        iftWriteImageByExt(ellipsesImage, filename);

        sprintf(filename, "ellipseMatch%d.png", i);
        iftWriteImageByExt(ellipseMatchImage, filename);

        iftDestroyVoxelArray(&centroids);
        iftDestroyImage(&mask);
        iftDestroyImage(&seeds);
        iftDestroyImage(&ellipsesImage);
        iftDestroyIGraph(&igraph);
        iftDestroyIGraph(&out_igraph);
    }

    mean_minor/=superpixel_counter;
    mean_major/=superpixel_counter;
    mean_anisotropy/=superpixel_counter;

    char filename[200];

    strcpy(filename, argv[3]);

    FILE *fp = fopen(filename,"w");

    fprintf(fp,"%f %f\n",mean_minor, mean_anisotropy);
    fprintf(fp,"%f %f\n",mean_major, mean_anisotropy);

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