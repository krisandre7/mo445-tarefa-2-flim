//
// Created by ilan on 05/12/2021.
//

#ifndef ROBOT_EXPERIMENTS_IFTDYNAMICFOREST_H
#define ROBOT_EXPERIMENTS_IFTDYNAMICFOREST_H

typedef struct ift_dyn_forest
{
    double    *cost;
    iftDHeap  *heap;
    iftImage  *label;
    iftImage  *marker;
    iftImage  *root;
    iftImage  *pred;
    iftMImage *mimg;
    iftAdjRel *A;

  iftDynamicList *dynamic_roots; // iftDynamicTree
  iftDynamicList **map_nodes;    // tree_map
} iftDynamicForest;


// iftAddMarkersToDynamicForest

void
iftDiffDynamicSeedsAddition(iftDynamicForest *forest, iftLabeledSet *seeds);

// iftRemMarkersToDynamicForest

void iftDiffDynamicSeedsRemoval(iftDynamicForest *forest, iftSet *removal_markers);


iftDynamicForest *iftCreateDynamicForest(iftMImage *mimg, iftAdjRel *A) {
    iftDynamicForest *forest = (iftDynamicForest*) iftAlloc(1, sizeof *forest);

    forest->cost           = iftAllocDoubleArray(mimg->n);
    forest->heap           = iftCreateDHeap(mimg->n, forest->cost);
    forest->label          = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    forest->marker         = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    forest->root           = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    forest->pred           = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);

    forest->dynamic_roots  = NULL;
    forest->map_nodes      = (iftDynamicList **) iftAlloc(mimg->n, sizeof (iftDynamicList*));

    forest->mimg           = mimg;
    forest->A              = A;

    for (ulong p = 0; p < mimg->n; p++)
    {
        forest->cost[p] = IFT_INFINITY_DBL;
        forest->pred->val[p] = IFT_NIL;
        forest->root->val[p] = p;
    }

    return forest;
}

void iftDestroyDynamicForest(iftDynamicForest **forest) {
    if ((*forest) != NULL) {
        iftDestroyDynamicList(&(*forest)->dynamic_roots);

        iftFree((*forest)->map_nodes);

        iftFree((*forest)->cost);
        iftDestroyDHeap(&(*forest)->heap);
        iftDestroyImage(&(*forest)->label);
        iftDestroyImage(&(*forest)->marker);
        iftDestroyImage(&(*forest)->root);
        iftDestroyImage(&(*forest)->pred);

        iftFree(*forest);
        *forest = NULL;
    }
}

// iftUpdateDynamicTree
void iftInsertDynamicList(iftDynamicForest *forest, int p) {
    int root             = forest->root->val[p];
    iftDynamicList *node = forest->map_nodes[root];
    iftMImage *mimg      = forest->mimg;

    iftInsertDynamicListVoxel(node, mimg->val[p]);

    // Um pequeno detalhe no código abaixo: quando eu conquisto um voxel eu preciso
    // saber se ele já foi conquistado por uma semente, pois se este for o caso, eu
    // removo da árvore associada à semente que conquistou antes. Assim,
    // map_nodes[p] é NULL se não houve conquista e map_nodes[p] != NULL se ele
    // já foi conquistado por uma semente.
    forest->map_nodes[p] = node;
}

// iftUpdateDynamicTree
void iftRemoveDynamicList(iftDynamicForest *forest, int p) {
    int r = forest->root->val[p];
    if (forest->map_nodes[r] && forest->map_nodes[p] == forest->map_nodes[r]) {
        iftDynamicList *node = forest->map_nodes[r];
        iftRemoveDynamicListVoxel(node, forest->mimg->val[p]);
        forest->map_nodes[p] = NULL;
    }
}

// iftDynamicArcWeight
double iftDistDynamicList(iftDynamicForest *forest, int p, int q) {
    iftMImage *mimg = forest->mimg;

    iftDynamicList *node = forest->map_nodes[forest->root->val[p]];

    double dist = iftDynamicListDistanceVoxel(node, mimg->val[q]);

    return dist;
}


iftSet *iftDynamicTreeRemoval(iftDynamicForest *fst, iftSet *trees_for_removal)
{
    int        i, p, q, r, n = fst->mimg->n;
    double V0;
    iftVoxel   u, v;
    iftAdjRel *A = fst->A;
    iftSet   *Frontier = NULL;
    iftBMap  *inFrontier = iftCreateBMap(n);
    iftImage  *pred = fst->pred, *root = fst->root;
    double *cost = fst->cost;
    iftMImage *mimg = fst->mimg;
    iftSet   *T1 = NULL, *T2 = NULL;

    if (fst->heap->removal_policy == MINVALUE)
        V0 = IFT_INFINITY_DBL;
    else // MAXVALUE
        V0 = IFT_INFINITY_DBL_NEG;

    /* Remove all marked trees and find the frontier voxels
       afterwards. */

    while (trees_for_removal != NULL) {
        p = trees_for_removal->elem;

        if (cost[root->val[p]] != V0) { //tree not marked yet
            r = root->val[p];
            cost[r] = V0;
            pred->val[r] = IFT_NIL;
            iftInsertSet(&T1, r);
            while (T1 != NULL) {
                p = iftRemoveSet(&T1);
                iftInsertSet(&T2, p);

                u = iftMGetVoxelCoord(mimg, p);
                for (i = 1; i < A->n; i++) {
                    v = iftGetAdjacentVoxel(A, u, i);
                    if (iftMValidVoxel(mimg, v)) {
                        q   = iftMGetVoxelIndex(mimg, v);
                        if (cost[q] != V0) {
                            if (pred->val[q] == p) {
                                iftInsertSet(&T1, q);
                                iftRemoveDynamicList(fst, q);
                                cost[q] = V0; // mark removed node
                                pred->val[q]    = IFT_NIL;
                            }
                        }
                    }
                }
            }
        }

        trees_for_removal = trees_for_removal->next;
    }

    /* Find the frontier voxels of non-removed trees */

    while (T2 != NULL)
    {
        p = iftRemoveSet(&T2);
        u = iftMGetVoxelCoord(mimg, p);
        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mimg, v))
            {
                q   = iftMGetVoxelIndex(mimg, v);
                if (cost[q] != V0)
                {
                    if (iftBMapValue(inFrontier, q) == 0)
                    {
                        iftInsertSet(&Frontier, q);
                        iftBMapSet1(inFrontier, q);
                    }
                }
            }
        }
    }
    iftDestroyBMap(&inFrontier);

    return (Frontier);
}

void iftDynamicSubtreeRemoval(iftDynamicForest *forest, int t) {
    iftMImage *mimg = forest->mimg;
    iftImage  *pred = forest->pred;
    iftImage  *label = forest->label;
    iftImage  *root = forest->root;
    double    *cost = forest->cost;
    iftAdjRel *A    = forest->A;
    iftDHeap  *heap = forest->heap;

    iftSet      *K = NULL;
    iftIntQueue *T = iftCreateIntQueue(mimg->n);
    iftIntArray *arr = iftCreateIntArray(mimg->n);

    iftInsertIntQueue(T, t);

    while (!iftIsIntQueueEmpty(T)) {
        int p;
        iftRemoveIntQueue(T, &p);

        if (heap->color[p] == IFT_GRAY) {
            iftRemoveDHeapElem(heap, p);
        }

        iftRemoveDynamicList(forest, p);
        heap->color[p]     = IFT_WHITE;
        pred->val[p]       = IFT_NIL;
        label->val[p]      = IFT_NIL;
        root->val[p]       = p;
        cost[p]            = IFT_INFINITY_DBL;

        iftVoxel u = iftMGetVoxelCoord(mimg, p);

        for (int i = 1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mimg, v)) {
                int q = iftMGetVoxelIndex(mimg, v);

                if (pred->val[q] == p) {
                    iftInsertIntQueue(T, q);
                } else {
                    if (arr->val[q] == 0 && cost[q] != IFT_INFINITY_DBL && root->val[q] != q) {
                        iftInsertSet(&K, q);
                        arr->val[q] = 1;
                    }
                }
            }
        }
    }

    while (K != NULL) {
        int p = iftRemoveSet(&K);
        arr->val[p] = 0;
        if (cost[p] != IFT_INFINITY_DBL && heap->color[p] != IFT_GRAY) {
            iftInsertDHeap(heap, p);
        }
    }

    iftDestroySet(&K);
    iftDestroyIntQueue(&T);
    iftDestroyIntArray(&arr);
}

// iftRemMarkersFromDynamicForest

void iftDiffDynamicSeedsRemoval(iftDynamicForest *forest, iftSet *removal_markers) {
    iftSet    *Frontier = NULL;
    iftDHeap *heap = forest->heap;

    if (removal_markers != NULL)
    {
        Frontier = iftDynamicTreeRemoval(forest, removal_markers);
        while (Frontier != NULL) {
            int p = iftRemoveSet(&Frontier);
            /* p is also a seed voxel, but the priority is it as a seed. */
            if (heap->color[p] != IFT_GRAY) {
                iftInsertDHeap(heap, p);
            }
        }
    }
}

// iftAddMarkersToDynamicForest

void iftDiffDynamicSeedsAddition(iftDynamicForest *forest, iftLabeledSet *seeds) {
    double    *cost = forest->cost;
    iftImage  *label = forest->label;
    iftImage  *pred = forest->pred;
    iftImage  *root = forest->root, *marker = forest->marker;
    iftDHeap *heap = forest->heap;

    iftLabeledSet *M = seeds;
    while (M != NULL)
    {
        int p = M->elem;

        if (heap->color[p] == IFT_GRAY) {
            /* p is also a frontier voxel, but the priority is it as a seed. */
            iftRemoveDHeapElem(heap, p);
        }

        cost[p]         = 0;
        label->val[p]   = M->label;
        root->val[p]    = p;
        pred->val[p]    = IFT_NIL;
        marker->val[p]  = M->marker;
        iftInsertDHeap(heap, p);
        forest->map_nodes[p] = iftCreateDynamicList(&forest->dynamic_roots, p, forest->mimg->m);
        iftInsertDynamicList(forest, p);

        M = M->next;
    }

}

#endif //ROBOT_EXPERIMENTS_IFTDYNAMICFOREST_H
