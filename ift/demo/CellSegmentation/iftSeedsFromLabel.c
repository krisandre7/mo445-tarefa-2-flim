#include "ift.h"

int main(int argc, char *argv[])
{
    iftImage        *label=NULL;
    char             ext[10],*pos;

    /*--------------------------------------------------------*/

    void *trash = malloc(1);
    struct mallinfo info;
    int MemDinInicial, MemDinFinal;
    free(trash);
    info = mallinfo();
    MemDinInicial = info.uordblks;

    /*--------------------------------------------------------*/


    if (argc<5)
        iftError("Usage: iftWaterGray <label.scn> <spatial_radius> <volume_thres> [basins.scn]","main");

    pos = strrchr(argv[1],'.') + 1;
    sscanf(pos,"%s",ext);

    if (strcmp(ext,"scn")==0){
        label   = iftReadImage(argv[1]);
    }else{
        printf("Invalid image format: %s\n",ext);
        exit(-1);
    }

    iftLabeledSet *S = iftGeodesicCenters(label);


    iftWriteSeeds("seeds.txt", S, label);

    iftDestroyImage(&label);
    iftDestroyLabeledSet(&S);

    /* ---------------------------------------------------------- */

    info = mallinfo();
    MemDinFinal = info.uordblks;
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

    return(0);
}

