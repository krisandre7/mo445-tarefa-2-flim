//
// Created by azaelmsousa on 09/09/20.
//

#include "ift.h"

int main(int argc, char *argv[])
{
    if (argc != 6)
        iftError("Usage iftCleanGGOSegmentation <...>\n"
                 "[1] GGO label image from segmentation network\n"
                 "[2] Lungs-Trachea segmentation image\n"
                 "[3] Border adjacency radius\n"
                 "[4] Opening radius\n"
                 "[5] Output GGO segmentation\n","main");

    timer *t1,*t2;

    iftImage *ggo = iftReadImageByExt(argv[1]);
    iftImage *segm = iftReadImageByExt(argv[2]);
    double adj_radius = atof(argv[3]);
    double close_radius = atof(argv[4]);

    printf("\n======================\n");
    printf("GGO label image: %s\n",argv[1]);
    printf("Lungs-Trachea segmentation: %s\n",argv[2]);
    printf("Border adjacency radius: %.2f\n",adj_radius);
    printf("Opening radius: %.2f\n",close_radius);
    printf("Output GGO segmentation: %s\n",argv[5]);
    printf("======================\n\n");

    if (!iftIsDomainEqual(ggo,segm))
        iftError("Input label images do not share the same space domain","main");

    t1 = iftTic();

    iftImage *segm_inner = iftCopyImage(segm);
    iftAdjRel *A = iftSpheric(adj_radius);

    puts("-- Extracting lungs segmentation outer border");
    for (int i = 0; i < segm->n; i++){
        if (segm->val[i] > 0) {
            iftVoxel u = iftGetVoxelCoord(segm,i);
            for (int j = 1; j < A->n; j++) {
                iftVoxel v = iftGetAdjacentVoxel(A,u,j);
                if (iftValidVoxel(segm,v)){
                    int p = iftGetVoxelIndex(segm,v);
                    if (segm->val[p] == 0){
                        segm_inner->val[i] = 1;
                        break;
                    }
                }
            }
        }
    }
    iftDestroyAdjRel(&A);
    iftDestroyImage(&segm);

    puts("-- Morphological Opening");
    iftImage *closed = iftOpenBin(ggo, close_radius);
    iftDestroyImage(&ggo);

    puts("-- Filtering images");
    for (int i = 0; i < segm_inner->n; i++){
        if (segm_inner->val[i] == 0){
            closed->val[i] = 0;
        }
    }
    iftDestroyImage(&segm_inner);

    iftWriteImageByExt(closed,argv[5]);
    iftDestroyImage(&closed);

    t2 = iftToc();

    puts("Done...");
    printf("Cleaning GGO Segmentation took %s\n",iftFormattedTime(iftCompTime(t1,t2)));

    return 0;
}