//
// Created by azaelmsousa on 24/10/20.
//

/* This program computes metrics regarding a ggo segmentation with its respective
 * ground truth for comparison purposes.
 */

#include "ift.h"

iftImage **iftReadImageSetFromDir(iftFileSet *fs)
{
    int first = 0;
    int last = fs->n - 1;

    iftImage **img = (iftImage **)calloc(fs->n, sizeof(iftMImage *));

    for (int i = first; i <= last; i++)
    {
        printf("Reading image: %d of %d\r", i + 1, (int)fs->n);
        fflush(stdout);
        img[i] = iftReadImageByExt(fs->files[i]->path);
    }
    printf("\n");

    return (img);
}

void iftDestroyImageSet(iftImage ***img, int nimages)
{
    iftImage **aux = *img;

    for (int i = 0; i < nimages; i++)
        iftDestroyImage(&aux[i]);
    free(aux);
    *img = NULL;
}

int main(int argc, char *argv[]) {

    if (argc != 4) {
        iftError("Usage: iftGGOMetrics <...>\n" \
                 "[1] Input dir with GGO segmentations\n" \
                 "[2] Dir with ground truth\n" \
                 "[3] Output csv\n", "main");
    }

    iftFileSet *fs_input   = iftLoadFileSetFromDirBySuffix(argv[1],"nii.gz", 1);
    iftFileSet *fs_gt      = iftLoadFileSetFromDirBySuffix(argv[2],"nii.gz", 1);

    iftImage **pred = iftReadImageSetFromDir(fs_input);
    iftImage **gt = iftReadImageSetFromDir(fs_gt);

    for (int i = 0; i < fs_input->n; i++) {
        printf("Processing image %03d/%03ld - ggo: %s | gt: %s\n", i + 1, fs_input->n, fs_input->files[i]->path,fs_gt->files[i]->path);

        if (!iftIsDomainEqual(gt[i], pred[i])) {
            iftWarning("Input images must have the same domain\n" \
                 "argv[1] = (%d,%d,%d), argv[2] = (%d,%d,%d)", "main",
                       pred[i]->xsize, pred[i]->ysize, pred[i]->zsize,
                       gt[i]->xsize, gt[i]->ysize, gt[i]->zsize);
            continue;
        }

        // Computing FPR and FNR

        iftAdjRel *A = iftSpheric(sqrt(3.));
        int n_tp = 0;
        int n_fn = 0;
        int n_fp = 0;

        iftImage *comp = iftLabelComp(gt[i], A);
        for (int i = 1; i <= iftMaximumValue(comp); i++) {
            iftImage *obj = iftExtractObject(comp, i);
            float size_obj = 0;
            float size_pred = 0;
            for (int p = 0; p < pred[i]->n; p++) {
                if (obj->val[p] > 0) {
                    size_obj++;
                    if (pred[i]->val[p] > 0)
                        size_pred++;
                }
            }
            if (size_pred / size_obj >= 0.5) {
                n_tp++;
            } else {
                n_fn++;
            }
            iftDestroyImage(&obj);
        }
        iftDestroyImage(&comp);

        comp = iftLabelComp(pred[i], A);
        for (int i = 1; i <= iftMaximumValue(comp); i++) {
            iftImage *obj = iftExtractObject(comp, i);
            float size_obj = 0;
            float size_gt = 0;
            for (int p = 0; p < gt[i]->n; p++) {
                if (obj->val[p] > 0) {
                    size_obj++;
                    if (gt[i]->val[p] > 0)
                        size_gt++;
                }
            }
            if (size_gt / size_obj < 0.5) {
                n_fp++;
            }
            iftDestroyImage(&obj);
        }
        iftDestroyImage(&comp);
        iftDestroyAdjRel(&A);


        float fnr = ((float) n_fn) / (n_fn + n_tp);

        for (int p = 0; p < pred[i]->n; p++) {
            if (pred[i]->val[p] > 0)
                pred[i]->val[p] = 1;
            if (gt[i]->val[p] > 0)
                gt[i]->val[p] = 1;
        }

        // Computing DICE
        double dice = iftDiceSimilarity(pred[i], gt[i]);

        // Computing ASSD
        double assd = iftASSD(pred[i], gt[i]);


        // Writing output file
        FILE *f;
        bool firstline = FALSE;
        if (!iftFileExists(argv[3]))
            firstline = TRUE;
        f = fopen(argv[3], "a+");
        if (f == NULL)
            iftError("Cannot open csv file", "main");
        if (firstline)
            fprintf(f, "patient;# fp;# fn;# tp;fnr;ggo dice;ggo assd\n");
        fprintf(f, "%s;%d;%d;%d;%f;%lf;%lf\n",
                iftFilename(argv[1], NULL), //patient
                n_fp,                       //number of false positive
                n_fn,                       //number of false negative
                n_tp,                       //number of true positive
                fnr,                        //false negative rate
                dice,                       //ggo dice
                assd);                      //ggo assd
        fclose(f);
    }
    iftDestroyImageSet(&pred,fs_input->n);
    iftDestroyImageSet(&gt,fs_gt->n);

    iftDestroyFileSet(&fs_input);
    iftDestroyFileSet(&fs_gt);

    return 1;
}