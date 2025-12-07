#include "ift.h"

iftImage *iftLargeMarginDynamicSegm(iftMImage *mimg, iftAdjRel *A, iftLabeledSet *seeds, int h,
                                    bool use_dist, int min_tree_size, int k_targets, double learn_rate, int iterations);

int main(int argc, char *argv[])
{
    if (argc != 9) {
        iftExit("iftLargeMarginDynamic <img path> <seeds path> <height> <min tree size> <k targets> <learning rate> <ite. between l.m.> <out path>", "iftLargeMarginDynamic");
    }

    const char *img_path = argv[1];
    const char *seeds_path = argv[2];
    int height = atoi(argv[3]);
    int min_tree_size = atoi(argv[4]);
    int k_targets = atoi(argv[5]);
    double learn_rate = atof(argv[6]);
    int iterations = atoi(argv[7]);
    const char *out_path = argv[8];

    iftImage *img = iftReadImageByExt(img_path);
    iftLabeledSet *seeds = iftReadSeeds(img, seeds_path);
    iftAdjRel *A = iftCircular(1.0f);

    iftMImage *mimg = iftImageToMImage(img, LABNorm_CSPACE);

    timer *tic = iftTic();
    iftImage *segm = iftLargeMarginDynamicSegm(mimg, A, seeds, height, false, min_tree_size, k_targets, learn_rate, iterations);
    timer *toc = iftToc();
    char *time = iftFormattedTime(iftCompTime(tic, toc));
    printf("%s\n", time);
    iftFree(time);

    iftImage *out = iftMask(img, segm);
    iftWriteImageByExt(out, out_path);
    iftDestroyImage(&out);
    iftDestroyImage(&segm);

    segm = iftDynamicSetRootPolicy(mimg, A, seeds, height, false);
    out = iftMask(img, segm);
    iftWriteImageByExt(out, "original.png");
    iftDestroyImage(&out);
    iftDestroyImage(&segm);

    iftDestroyAdjRel(&A);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
    iftDestroyLabeledSet(&seeds);

    return 0;
}


double iftDistMahalonobisDynSet(iftDynamicSet *S,   iftMImage *mimg,   double *M, int q)
{
    double *diff = iftAlloc(mimg->m, sizeof *diff);
    double *aux = iftAlloc(mimg->m, sizeof *aux);

    for (int i = 0; i < mimg->m; i++) {
        aux[i] = 0;
        diff[i] = S->mean[i] - mimg->val[q][i];
    }

    for (int i = 0; i < mimg->m; i++) {
        for (int j = 0; j < mimg->m; j++) {
            aux[i] += diff[j] * M[j * mimg->m + i];
        }
    }

    double out = 0;
    for (int i = 0; i < mimg->m; i++) {
        out += aux[i] * diff[i];
    }

    iftFree(diff);
    iftFree(aux);

    return out;
}


void iftDynamicSetToPointers(  iftLabeledSet *seeds, iftDynamicSet **S, int min_nodes, double **data_,
                             int **label_, int* size, int **pixel_)
{
    *size = 0;
    int d = 0;
    for (  iftLabeledSet *C = seeds; C != NULL; C = C->next) {
        int i = C->elem;
        if (S[i] != NULL && S[i]->size >= min_nodes) {
            d = S[i]->dim;
            (*size)++;
        }
    }


    *data_ = iftAlloc((*size) * d, sizeof (**data_));
    *label_ = iftAlloc((*size), sizeof (**label_));
    *pixel_ = iftAlloc((*size), sizeof (**pixel_));

    double *data = *data_;
    int *label = *label_;
    int *pixel = *pixel_;

    int s = 0;
    for (  iftLabeledSet *C = seeds; C != NULL; C = C->next) {
        int i = C->elem;
        if (S[i] != NULL && S[i]->size > min_nodes)
        {
            for (int j = 0; j < d; j++)
            {
                data[s * d + j] = S[i]->mean[j];
            }
            label[s] = C->label;
            pixel[s] = i;
            s++;
        }
    }
}


iftImage *iftLargeMarginDynamicSegm(iftMImage *mimg, iftAdjRel *A, iftLabeledSet *seeds, int h,
        bool use_dist, int min_tree_size, int k_targets, double learn_rate, int ite_between)
{
    iftImage *label   = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    iftFImage *pathval = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
    iftImage *root = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);

    iftDynamicSet **S = iftAlloc(mimg->n, sizeof (*S));

    iftFHeap *H = iftCreateFHeap(mimg->n, pathval->val);

    for (int p = 0; p < mimg->n; p++)
    {
        pathval->val[p] = IFT_INFINITY_FLT;
        root->val[p] = IFT_NIL;
    }

    for (iftLabeledSet *M = seeds; M != NULL; M = M->next)
    {
        int p = M->elem;
        label->val[p] = M->label;
        pathval->val[p] = h;
        root->val[p] = p;
        iftInsertFHeap(H, p);
        S[p] = iftCreateDynamicSet(mimg->m);
        iftInsertDynamicSet(S[p], mimg, p);
    }

    double *X = NULL;
    int *y = NULL;
    int *pixels = NULL;
    int size = -1;
    iftDynamicSetToPointers(seeds, S, 0, &X, &y, &size, &pixels);
    double *L = iftComputeLargeMargin(X, y, NULL, size, mimg->m, k_targets, learn_rate, 1000, true);
    double *M = iftSpaceTransform(L, L, mimg->m, mimg->m);

    int ite = 0;
    while (!iftEmptyFHeap(H))
    {
        ite++;
        int p = iftRemoveFHeap(H);
        iftVoxel u = iftMGetVoxelCoord(mimg, p);

        if (root->val[p] == p)
        {
            pathval->val[p] = 0;
        }

        if (p != root->val[p]) { /* roots have already been inserted */
            iftInsertDynamicSet(S[root->val[p]], mimg, p);
        }

        if (ite / ite_between == 0)
        {
            iftFree(X);
            iftFree(y);
            iftFree(pixels);
            iftFree(M);
            double *aux_L = L;
            iftDynamicSetToPointers(seeds, S, min_tree_size, &X, &y, &size, &pixels);
            L = iftComputeLargeMargin(X, y, aux_L, size, mimg->m, k_targets, learn_rate, 1000, true);
            iftFree(aux_L);
            M = iftSpaceTransform(L, L, mimg->m, mimg->m);
        }

        for (int i = 1; i < A->n; i++)
        {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if (iftMValidVoxel(mimg, v))
            {
                int q = iftMGetVoxelIndex(mimg, v);

                if (H->color[q] != IFT_BLACK) {
                    double arc_weight = iftDistMahalonobisDynSet(S[root->val[p]], mimg, M, q);
                    double tmp;
                    if (use_dist) {
                        tmp = iftMax(pathval->val[p], arc_weight + iftMImageMahalanobisDouble(mimg, M, p, q));
//                        tmp = arc_weight + iftMImageMahalanobisDouble(mimg, M, p, q);
                    } else {
                        tmp = iftMax(pathval->val[p], arc_weight);
//                        tmp = arc_weight;
                    }

                    if (tmp < pathval->val[q])
                    {
                        label->val[q] = label->val[p];
                        pathval->val[q] = (float) tmp;
                        root->val[q] = root->val[p];

                        if (H->color[q] == IFT_GRAY)
                            iftGoUpFHeap(H, H->pos[q]);
                        else
                            iftInsertFHeap(H, q);
                    }
                }
            }
        }
    }

    iftDestroyFImage(&pathval);
    iftDestroyImage(&root);
    for (iftLabeledSet *C = seeds; C != NULL; C = C->next) {
        int p = C->elem;
        if (S[p] != NULL) {
            iftDestroyDynamicSet(&S[p]);
        }
    }
    iftFree(S);
    iftDestroyFHeap(&H);
    iftFree(X);
    iftFree(y);
    iftFree(M);
    iftFree(L);

    return label;
}


