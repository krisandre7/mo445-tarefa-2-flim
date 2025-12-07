#include "ift.h"
#include "iftSaliency.h"

void iftNormalizeBandsInPlace(iftMImage *mimg)
{
    for (int b=0; b < mimg->m; b++) {
        float min = IFT_INFINITY_FLT;
        float max = IFT_INFINITY_FLT_NEG;
        for (int p = 0; p < mimg->n; p++) {
            if (mimg->val[p][b] < min)
                min = mimg->val[p][b];
            if (mimg->val[p][b] > max)
                max = mimg->val[p][b];
        }
        if (!iftAlmostZero(max-min))
            for (int p = 0; p < mimg->n; p++) {
                mimg->val[p][b] = (mimg->val[p][b]-min)/(max-min);
            }
    }
}

iftImage *iftRemoveComponentsWithoutScribbles(iftImage *segmented, iftImage *scribbles){
    iftImage *out_label_img = iftCreateImageFromImage(segmented);

    iftIntQueue *Q = iftCreateIntQueue(segmented->n);
    iftAdjRel *A = iftSpheric((float)sqrt(3.0));

    int label = 1;

    for (int t = 0; t < segmented->n; t++) {
        if ((segmented->val[t] != 0) && (out_label_img->val[t] == 0)) {
            out_label_img->val[t] = label;
            iftInsertIntQueue(Q, t);

            while (!iftIsIntQueueEmpty(Q)) {
                int p;
                iftRemoveIntQueue(Q, &p);
                iftVoxel u = iftGetVoxelCoord(segmented, p);

                for (int i = 1; i < A->n; i++) {
                    iftVoxel v = {.x = u.x + A->dx[i], .y = u.y + A->dy[i], .z = u.z + A->dz[i]};

                    if (iftValidVoxel(segmented, v)) {
                        int q = iftGetVoxelIndex(segmented, v);

                        if ((out_label_img->val[q] == 0) && (segmented->val[p] == segmented->val[q])) {
                            out_label_img->val[q] = label;
                            iftInsertIntQueue(Q, q);
                        }
                    }
                }
            }
            label++;
        }
    }


    iftImage *segmented_scribble_only = iftCreateImageFromImage(segmented);
    iftIntArray *label_with_scribbles = iftCreateIntArray(label);
    for(int p = 0; p < segmented->n; p++)
        if(scribbles->val[p] > 1)
            label_with_scribbles->val[out_label_img->val[p]] = 1;
        else if(scribbles->val[p] == 1)
            label_with_scribbles->val[out_label_img->val[p]] = 0;


    for(int p = 0; p < segmented->n; p++)
        if(label_with_scribbles->val[out_label_img->val[p]] == 1)
            segmented_scribble_only->val[p] = 255;

    iftDestroyIntQueue(&Q);
    iftDestroyAdjRel(&A);
    iftDestroyIntArray(&label_with_scribbles);

    iftDestroyImage(&out_label_img);
    return segmented_scribble_only;
}

int main(int argc, const char *argv[]) {
    /* General variables */
    char orig_file[250], scribbles_file[250], output_file[250], features_file[250];
    iftImage *orig, *initial_sal, *final_sal;
    iftMImage *features = NULL;
    int limit_connection = 0, start_superpixel_number;

    if (argc != 7 && argc != 6) {
        iftError("Usage: iftSESS <orig_image> <scribbles_file> <output_saliency> <limit_connection_to_scribbles>(0,1) <number_of_superpixels> |OPTIONAL|<features.mimg>", "main");
    }

    strcpy(orig_file, argv[1]);
    strcpy(scribbles_file, argv[2]);
    strcpy(output_file, argv[3]);
    limit_connection = atoi(argv[4]);
    start_superpixel_number = atoi(argv[5]);
    if(argc == 7) {
        strcpy(features_file, argv[6]);
        features = iftReadMImage(features_file);
        iftNormalizeBandsInPlace(features);
    }

    orig = iftReadImageByExt(orig_file);
    iftLabeledSet *scribbles = iftReadSeeds(orig, scribbles_file);
    initial_sal = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
    iftLabeledSetToImage(scribbles, initial_sal, 1); //background = 1, foreground = 2
    iftITSELFParameters *params = iftInitializeITSELFParametersByDefaultScribbles();
    params->number_superpixels = start_superpixel_number;

    iftAdjRel       *A           = iftCircular(2.0);
    iftMImage *mimg = iftImageToMImage(orig, YCbCr_CSPACE);
    iftImage *superpixels = iftDISF(mimg, A, 8000, 2000, NULL);
    iftDestroyMImage(&mimg);
    iftDestroyAdjRel(&A);
    iftIntArray *superpixel_has_scribble = iftCreateIntArray(start_superpixel_number);

    for(int p = 0; p < superpixels->n; p++)
        if(initial_sal->val[p] == 2)
            superpixel_has_scribble->val[superpixels->val[p]] += 1;

    for(int p = 0; p < initial_sal->n; p++)
        if(superpixel_has_scribble->val[superpixels->val[p]] > 10)
            initial_sal->val[p] = 255;

    iftDestroyImage(&superpixels);
    iftDestroyIntArray(&superpixel_has_scribble);

    iftWriteImageByExt(initial_sal, "ini.png");

    final_sal = iftSESS(orig, initial_sal, params, features);

    iftDestroyMImage(&features);

    if(limit_connection) {
        int threshold = 30;
        iftImage *segmented = iftCreateImageFromImage(final_sal);
        for (int p = 0; p < final_sal->n; p++)
            if(initial_sal->val[p] == 1)
                segmented->val[p] = 0;
            else
                segmented->val[p] = final_sal->val[p] >= threshold;
        iftImage *segmented_only_scribbles = iftRemoveComponentsWithoutScribbles(segmented, initial_sal);
        for (int p = 0; p < final_sal->n; p++)
            if (segmented_only_scribbles->val[p] == 0)
                final_sal->val[p] = 0;
    }

    iftDestroyITSELFParameters(&params);

    iftWriteImageByExt(final_sal, output_file);


    iftDestroyImage(&orig);
    iftDestroyImage(&initial_sal);
    iftDestroyLabeledSet(&scribbles);
    iftDestroyImage(&final_sal);

    return 0;
}