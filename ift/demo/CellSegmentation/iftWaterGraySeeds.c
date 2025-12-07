#include "ift.h"

int main(int argc, char *argv[])
{
    iftImageForest        *fst = NULL;
    iftImage        *img=NULL,*basins = NULL;
    iftImage        *marker=NULL;
    iftAdjRel       *A=NULL;
    char             ext[10],*pos;
    timer           *t1=NULL,*t2=NULL;

    /*--------------------------------------------------------*/

    void *trash = malloc(1);
    struct mallinfo info;
    int MemDinInicial, MemDinFinal;
    free(trash);
    info = mallinfo();
    MemDinInicial = info.uordblks;

    /*--------------------------------------------------------*/


    if (argc<5)
        iftError("Usage: iftWaterGray <image.scn> <spatial_radius> <volume_thres> [basins.scn]","main");

    pos = strrchr(argv[1],'.') + 1;
    sscanf(pos,"%s",ext);

    if (strcmp(ext,"scn")==0){
        img   = iftReadImage(argv[1]);
    }else{
        printf("Invalid image format: %s\n",ext);
        exit(-1);
    }

    t1 = iftTic();

    /* the operation is connected for the topology defined by A: A must
       be the same in all operators (including
       iftVolumeClose?). Otherwise, this is not a connected operation in
       the desired topology. */

    A      = iftSpheric(atof(argv[2]));
    if(argc >= 6) {
        basins = iftReadImage(argv[5]);
    } else {
        basins = iftImageBasins(img, A);
    }

    iftWriteImage(basins,"basins.scn");

    fst = iftCreateImageForest(basins, A);

    marker = iftVolumeClose(basins,atof(argv[3]));
    iftWaterGrayForest(fst,marker);

    iftLabeledSet *S = iftGeodesicCenters(fst->label);

    t2     = iftToc();

    fprintf(stdout,"watergray in %f ms with %d regions\n",iftCompTime(t1,t2),iftMaximumValue(fst->label));

    iftWriteImage(fst->label,"result.scn");
    iftWriteSeeds("seeds.txt", S, fst->label);

    iftDestroyAdjRel(&A);
    iftDestroyImage(&img);
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyImageForest(&fst);
    iftDestroyLabeledSet(&S);

    /* ---------------------------------------------------------- */

    info = mallinfo();
    MemDinFinal = info.uordblks;
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

    return(0);
}

