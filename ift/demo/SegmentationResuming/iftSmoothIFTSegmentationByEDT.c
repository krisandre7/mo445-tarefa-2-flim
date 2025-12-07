#include "ift.h"

iftLabeledSet *iftTieZoneSeeds(iftImage *cost, iftImage *label, iftAdjRel *A, iftSet **tiezone) {
    iftFIFO *fifo = iftCreateFIFO(cost->n);
    iftSet *frontier = NULL;
    iftLabeledSet *seeds    = NULL;
    iftBMap *inserted = iftCreateBMap(cost->n);

    if(tiezone != NULL) *tiezone = NULL;

    for(int p = 0; p < cost->n; p++) {
        iftVoxel u = iftGetVoxelCoord(cost, p);

        for(int i = 1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if(iftValidVoxel(cost, v)) {
                int q = iftGetVoxelIndex(cost, v);

                if(label->val[p] != label->val[q]) {
                    if(cost->val[p] == cost->val[q]) {
                        if(!iftBMapValue(inserted, p)) {
                            iftInsertSet(&frontier, p);
                            iftBMapSet1(inserted, p);

                        }

                        if(!iftBMapValue(inserted, q)) {
                            iftInsertSet(&frontier, q);
                            iftBMapSet1(inserted, q);
                        }
                    }
                }
            }
        }
    }

    iftFillBMap(inserted, 0);

    if(frontier != NULL) {
        while(frontier != NULL) {
            int p = iftRemoveSet(&frontier);
            iftInsertFIFO(fifo, p);
            iftBMapSet1(inserted, p);
            if(tiezone != NULL) iftInsertSet(tiezone, p);
        }

        while (!iftEmptyFIFO(fifo)) {
            int p = iftRemoveFIFO(fifo);
            iftVoxel u = iftGetVoxelCoord(cost, p);

            for(int i = 1; i < A->n; i++) {
                iftVoxel v = iftGetAdjacentVoxel(A, u, i);

                if (iftValidVoxel(cost, v)) {
                    int q = iftGetVoxelIndex(cost, v);
                    if (cost->val[p] == cost->val[q]) {
                        if(!iftBMapValue(inserted, q)) {
                            iftInsertFIFO(fifo, q);
                            iftBMapSet1(inserted, q);
                            if(tiezone != NULL) iftInsertSet(tiezone, q);
                        }
                    } else if(!iftBMapValue(inserted, q)){
                        iftInsertLabeledSet(&seeds, q, label->val[q]);
                        iftBMapSet1(inserted, q);
                    }
                }
            }
        }
    }



    iftDestroyFIFO(&fifo);
    iftDestroyBMap(&inserted);

    return seeds;
}

iftImage *iftIFTWithEDTPathCostOnTieZone(iftImage *label, iftLabeledSet *seeds, iftSet *tiezone, iftAdjRel *A) {
    iftImage *cost = NULL;
    iftImage *root = NULL;
    iftImage *smoothed = NULL;
    iftGQueue *Q = NULL;

    cost     = iftCreateImage(label->xsize, label->ysize, label->zsize);
    root     = iftCreateImage(label->xsize, label->ysize, label->zsize);
    Q        = iftCreateGQueue(QSIZE, label->n, cost->val);
    smoothed = iftCopyImage(label);

    iftSetImage(cost, IFT_INFINITY_INT_NEG);

    for(iftSet *S = tiezone; S != NULL; S = S->next) {
        cost->val[S->elem] = IFT_INFINITY_INT;
    }

    for(iftLabeledSet *S = seeds; S != NULL; S = S->next) {
        int p = S->elem;
        cost->val[p] = 0;
        root->val[p] = p;
        iftInsertGQueue(&Q, p);
    }


    // Image Foresting Transform

    while(!iftEmptyGQueue(Q)) {
        int p=iftRemoveGQueue(Q);

        iftVoxel u = iftGetVoxelCoord(label,p);
        iftVoxel r = iftGetVoxelCoord(label,root->val[p]);

        for (int i=1; i < A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A,u,i);
            if (iftValidVoxel(label,v)){
                int q = iftGetVoxelIndex(label,v);
                if (cost->val[q] > cost->val[p]){
                    int tmp = (v.x-r.x)*(v.x-r.x) + (v.y-r.y)*(v.y-r.y) + (v.z-r.z)*(v.z-r.z);
                    if (tmp < cost->val[q]){
                        if (cost->val[q] != IFT_INFINITY_INT)
                            iftRemoveGQueueElem(Q, q);
                        cost->val[q]  = tmp;
                        root->val[q]  = root->val[p];
                        smoothed->val[q]  = smoothed->val[p];
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    for(int p = 0; p < cost->n; p++) {
        cost->val[p] = iftMax(cost->val[p], 0);
    }

    iftDestroyImage(&root);
    iftDestroyImage(&cost);
    iftDestroyGQueue(&Q);

    return  smoothed;
}

int main(int argc, const char *argv[]) {
    iftImage *cost = NULL;
    iftImage *label = NULL;
    iftImage *smoothed = NULL;
    iftAdjRel *A = NULL;

    if(argc != 5)
        iftError("usage: %s <volume.scn/.pgm/.png> <label.scn/.pgm> <adjacency radius> <output label.scn/.pgm>", "main", argv[0]);

    cost = iftReadImageByExt(argv[1]);
    label = iftReadImageByExt(argv[2]);

    if(iftIs3DImage(cost)) {
        A = iftSpheric(atof(argv[3]));
    } else {
        A = iftCircular(atof(argv[3]));
    }

    iftSet *tiezone = NULL;
    iftLabeledSet *seeds = NULL;

    seeds = iftTieZoneSeeds(cost, label, A, &tiezone);



    smoothed = iftIFTWithEDTPathCostOnTieZone(label, seeds, tiezone, A);

    iftWriteImageByExt(smoothed, argv[4]);

    iftAdjRel *B = iftCircular(0.0);
    iftImage *cpy = iftCopyImage(label);
    iftDrawPoints(cpy, tiezone, iftRGBtoYCbCr(iftRGBColor(255,0,0), 255), B);
    if(iftIs3DImage(cpy)) {
        iftWriteImageByExt(cpy, "tiezone.scn",);
    } else {
        iftWriteImageByExt(cpy, "tiezone.png",);
    }
    iftDestroyImage(&cpy);

    cpy = iftCopyImage(label);
    iftSet *tmp = iftLabeledSetElemsToSet(seeds);
    iftDrawPoints(cpy, tmp, iftRGBtoYCbCr(iftRGBColor(255,0,0), 255), B);
    if(iftIs3DImage(cpy)) {
        iftWriteImageByExt(cpy, "seeds.scn",);
    } else {
        iftWriteImageByExt(cpy, "seeds.png",);
    }
    iftDestroyImage(&cpy);
    iftDestroySet(&tmp);

    iftDestroyLabeledSet(&seeds);
    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
    iftDestroySet(&tiezone);
    iftDestroyImage(&cost);
    iftDestroyImage(&label);
    iftDestroyImage(&smoothed);

    return  0;
}
