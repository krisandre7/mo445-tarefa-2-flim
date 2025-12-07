#include "iftMImage.h"
#include "iftAdjacency.h"
#include "ift/core/dtypes/LabeledSet.h"
#include "iftImageSequence.h"
#include "myDynTree.h"
#include "myArcWeightFunctions.h"

myDynTrees *myCreateDynTrees(iftMImage *img, iftAdjRel *A, iftLabeledSet **S, ArcWEntry *entry) {
    myDynTrees *dt = (myDynTrees *)calloc(1, sizeof(myDynTrees));

    dt->img = img;
    dt->A = A;
    dt->label = iftCreateImage(img->xsize, img->ysize, img->zsize);
    dt->root = iftCreateImage(img->xsize, img->ysize, img->zsize);
    dt->cost = iftCreateImage(img->xsize, img->ysize, img->zsize);
    dt->pred = iftCreateImage(img->xsize, img->ysize, img->zsize);
    dt->Q = iftCreateGQueue(CMAX + 1, img->n, dt->cost->val);
    dt->arcwfn = entry->fn;

    // allocate cumulative feature and treesize arrays
    if (entry->id == 1 || entry->id == 4) {
        dt->cumfeat = iftCreateMImage(img->xsize, img->ysize, img->zsize, img->m);
        dt->treesize = iftAllocIntArray(img->n);
    }

    // allocate rolling label sum array
    if (entry->id == 3 || entry->id == 6) {
        int max_label = 0;
        if (S != NULL && *S != NULL) {
            for (iftLabeledSet *aux = *S; aux != NULL; aux = aux->next) {
                if (aux->label > max_label) max_label = aux->label;
            }
        }
        dt->max_label = max_label;
        dt->label_size = iftAllocIntArray(max_label+1);
        dt->label_sum = (int**)calloc(dt->max_label+1, sizeof(int*));
        for (int l=0; l<=dt->max_label; l++) {
            dt->label_sum[l] = (int*)calloc(img->m, sizeof(int));
        }
        dt->tmp_mean = (float*)calloc(img->m, sizeof(float));
    }

    // allocate root cache
    if (entry->id == 2 || entry->id == 5) {
        dt->nroots = 0;
        dt->rootcache = NULL;
        if (S != NULL && *S != NULL) {
            // count seeds
            int count=0;
            for (iftLabeledSet *aux=*S; aux!=NULL; aux=aux->next) count++;
            dt->rootcache = (CachedRoot*)calloc(count, sizeof(CachedRoot));
            dt->nroots = count;
            int i=0;
            for (iftLabeledSet *aux=*S; aux!=NULL; aux=aux->next, i++) {
                dt->rootcache[i].r = aux->elem;
                dt->rootcache[i].label = aux->label;
                dt->rootcache[i].mean = (float*)calloc(img->m, sizeof(float));
                dt->rootcache[i].size = 0;
            }
        }
    }

    /* initialize maps */
    for (ulong p = 0; p < img->n; p++) {
        dt->cost->val[p] = IFT_INFINITY_INT;
        dt->pred->val[p] = IFT_NIL;
    }

    return (dt);
}

void myDestroyDynTrees(myDynTrees **dt) {
    myDynTrees *aux = *dt;

    if (aux != NULL) {
        iftDestroyGQueue(&aux->Q);
        iftDestroyImage(&aux->cost);
        iftDestroyImage(&aux->pred);
        iftDestroyImage(&aux->label);
        iftDestroyImage(&aux->root);

        if (aux->tmp_mean)
            free(aux->tmp_mean);

        if (aux->cumfeat)
            iftDestroyMImage(&aux->cumfeat);
        if (aux->treesize)
            iftFree(aux->treesize);

        if (aux->rootcache) {
            for (int i=0;i<aux->nroots;i++)
                if (aux->rootcache[i].mean) free(aux->rootcache[i].mean);
            free(aux->rootcache);
        }
        if (aux->label_size)
            iftFree(aux->label_size);

        if (aux->label_sum) {
            for (int l = 0; l <= aux->max_label; l++) {
                if (aux->label_sum[l]) free(aux->label_sum[l]);
            }
            free(aux->label_sum);
        }

        iftFree(aux);

        *dt = NULL;
    }
}

void myExecDynTrees(myDynTrees *dt, iftLabeledSet **S) {
    iftMImage *img = dt->img;
    iftAdjRel *A = dt->A;
    iftImage *cost = dt->cost;
    iftImage *label = dt->label;
    iftImage *pred = dt->pred;
    iftImage *root = dt->root;
    iftGQueue *Q = dt->Q;
    iftMImage *cumfeat = dt->cumfeat;
    int *treesize = dt->treesize;

    while (*S != NULL) {
        int lambda;
        int p = iftRemoveLabeledSet(S, &lambda);
        cost->val[p] = 0;
        label->val[p] = lambda;
        pred->val[p] = IFT_NIL;
        root->val[p] = p;

        if (dt->arcwfn == iftArcWeight1 || dt->arcwfn == iftArcWeight4) {
            for (ulong b = 0; b < img->m; b++) cumfeat->val[p][b] = 0;
            treesize[p] = 0;
        }
        iftInsertGQueue(&Q, p);
    }

    /* Execute the Image Foresting Transform of DT */

    while (!iftEmptyGQueue(Q)) {
        int p = iftRemoveGQueue(Q);
        iftVoxel u = iftMGetVoxelCoord(img, p);

        /* set / update dynamic tree of p */
        int r = root->val[p];
        if (dt->arcwfn == iftArcWeight1 || dt->arcwfn == iftArcWeight4) {
            for (ulong b = 0; b < img->m; b++) cumfeat->val[r][b] += img->val[p][b];
            treesize[r] += 1;
        }

        // also update rootcache mean incrementally if p is a root
        if (dt->arcwfn == iftArcWeight2 || dt->arcwfn == iftArcWeight5) {
            for (int i=0;i<dt->nroots;i++) {
                if (dt->rootcache[i].r == r) {
                    int n = ++dt->rootcache[i].size;
                    for (ulong b=0; b<img->m; b++) {
                        dt->rootcache[i].mean[b] =
                            ((dt->rootcache[i].mean[b]*(n-1)) + img->val[p][b]) / n;
                    }
                    break;
                }
            }
        }

        // update rolling sum and count for w3
        if (dt->arcwfn == iftArcWeight3 || dt->arcwfn == iftArcWeight6) {
            int lbl = label->val[p];
            dt->label_size[lbl] += 1;
            for (ulong b=0; b<img->m; b++) {
                dt->label_sum[lbl][b] += img->val[p][b];
            }
        }

        /* visit the adjacent voxels for possible path extension */
        for (int i = 1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if (iftMValidVoxel(img, v)) {
                int q = iftMGetVoxelIndex(img, v);
                if (Q->L.elem[q].color != IFT_BLACK) {
                    float arcw = dt->arcwfn(p, q, r, dt);
                    int tmp = iftMax(cost->val[p], iftMin(arcw, CMAX));
                    if (tmp < cost->val[q]) {
                        if (Q->L.elem[q].color == IFT_GRAY)
                            iftRemoveGQueueElem(Q, q);
                        cost->val[q] = tmp;
                        label->val[q] = label->val[p];
                        root->val[q] = root->val[p];
                        pred->val[q] = p;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    iftResetGQueue(Q);
}

iftImage *myDynamicTree(iftMImage *img, iftLabeledSet *S, int arcw_id) {
    ArcWEntry *entry = getArcWEntry(arcw_id);
    if (entry == NULL) {
        fprintf(stderr, "Error: Arc Weight Function with id %d not found!\n", arcw_id);
        exit(1);
    }

    iftAdjRel *A;

    if (iftIs3DMImage(img))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.0);

    // Split seeds
    iftLabeledSet *So = NULL, *Sb = NULL;
    for (iftLabeledSet *aux = S; aux != NULL; aux = aux->next) {
        if (aux->label == 1)
            iftInsertLabeledSet(&So, aux->elem, aux->label);
        else
            iftInsertLabeledSet(&Sb, aux->elem, aux->label);
    }

    // Copy seeds for dynamic tree use
    iftLabeledSet *tempS = iftCopyLabeledSet(S);
    myDynTrees *dt = myCreateDynTrees(img, A, &tempS, entry);

    iftLabeledSet *Scopy = iftCopyLabeledSet(S);

    // Measure execution time around myExecDynTrees
    myExecDynTrees(dt, &Scopy);

    iftImage *label = iftCopyImage(dt->label);
    myDestroyDynTrees(&dt);

    return label;
}

iftImage *myDynamicTreeInOut(iftMImage *img,
                             iftImage *seeds_in,
                             iftImage *seeds_out,
                             int arcw_id)
{
    ArcWEntry *entry = getArcWEntry(arcw_id);
    if (entry == NULL) {
        fprintf(stderr, "Error: Arc Weight Function with id %d not found!\n", arcw_id);
        exit(1);
    }

    iftAdjRel *A = iftIs3DMImage(img) ? iftSpheric(1.0) : iftCircular(1.0);

    // Build unified labeled seed set S from seeds_in and seeds_out
    iftLabeledSet *S = NULL;
    for (int p = 0; p < img->n; p++) {
        if (seeds_in->val[p] != 0) {
            // object/internal seed; you may keep seeds_in->val[p] if multi-label
            iftInsertLabeledSet(&S, p, seeds_in->val[p]); 
        } else if (seeds_out->val[p] != 0) {
            // external/background seed, label 0
            iftInsertLabeledSet(&S, p, 0);
        }
    }

    iftLabeledSet *tempS = iftCopyLabeledSet(S);
    myDynTrees *dt = myCreateDynTrees(img, A, &tempS, entry);

    iftLabeledSet *Scopy = iftCopyLabeledSet(S);
    myExecDynTrees(dt, &Scopy);

    iftImage *label = iftCopyImage(dt->label);

    iftDestroyAdjRel(&A);
    iftDestroyLabeledSet(&S);       // if you have a destroy function
    myDestroyDynTrees(&dt);

    return label;
}