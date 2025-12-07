#include "iftSaliency.h"
#include "iftSaliencyPriors.h"

iftImage **readImageSet(iftFileSet *fs);

int main(int argc, const char *argv[]) {

    if (argc != 3 && argc!=4) {
        iftError("Usage: iftEllipseMatch <bin_path> <out_path> <save_best_ellipse> (optional, boolean)", "main");
    }

    char out_path[250];
    iftFileSet *fs_binary = iftLoadFileSetFromDirOrCSV(argv[1], 1, 1);

    strcpy(out_path, argv[2]);
    iftImage **gt = readImageSet(fs_binary);

    int save_best_fit_ellipse = 0;
    if(argc==4)
        save_best_fit_ellipse = 1;

//    iftWriteImageByExt(gt[0], "test.png");
    int nimages = fs_binary->n;

    /* CHANGE THIS FOR IF YOU ARE PROVIDING A MAP WITH MULTIPLE LABELS) */
    for(int i = 0; i < nimages; i++)
        for(int p = 0; p < gt[i]->n; p++) {
            if(gt[i]->val[p] > 0)
                gt[i]->val[p] = 2;
            else
                gt[i]->val[p] = 1;
        }

    for(int i = 0; i < nimages; i++){
        iftImage *mask = iftSelectImageDomain(gt[i]->xsize, gt[i]->ysize, gt[i]->zsize);
        for(int p = 0; p < mask->n; p++) {
            if(mask->val[p] > 0)
                mask->val[p] = 2;
            else
                mask->val[p] = 1;
        }
        iftImage *label_img = gt[i];
        iftVoxelArray *centroids = iftGeometricCentersFromLabelImage(label_img);

        iftTensorScale *tensor_scale = iftSuperpixelToTensorScale(label_img, 32, 0, 100000);
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

        for(int p = 0; p < ellipseMatch->n; p++)
            ellipseMatch->val[p] = 1 * (ellipseMatch->val[p] - min) / (max - min);

        iftImage *ellipseMatchImage = iftCreateImageFromImage(label_img);
        for(int p = 0; p < label_img->n; p++)
            ellipseMatchImage->val[p] = (int)(255 * ellipseMatch->val[label_img->val[p]-1]);
        char filename[200];

        char *basename = iftFilename(fs_binary->files[i]->path, iftFileExt(fs_binary->files[i]->path));
        sprintf(filename, "%s/%s.png", out_path, basename);
        iftWriteImageByExt(ellipseMatchImage, filename);

        if(save_best_fit_ellipse == 1) {
            sprintf(filename, "%s/%s-ellipse.png", out_path, basename);
            iftWriteImageByExt(ellipsesImage, filename);
        }

        iftDestroyVoxelArray(&centroids);
        iftDestroyImage(&mask);
        iftDestroyImage(&ellipsesImage);
    }
    iftDestroyFileSet(&fs_binary);

    for(int i = 0; i < nimages; i++){
        iftDestroyImage(&gt[i]);
        iftDestroyImage(&gt[i]);
    }

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
