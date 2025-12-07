//
// Created by azaelmsousa on 23/03/21.
//

#include "ift.h"

int main(int argc, char *argv[])
{

    if (argc != 4){
        iftError("Usage iftNormalLungsReferenceSpaces <...>\n[1] Folder containing normal aligned images.\n[2] Number of clusters.\n[3] Output clusterized dataset.","main");
    }

    /* Variable initialization */

    iftFileSet *fs = iftLoadFileSetFromDirOrCSV(argv[1],1,0);
    int nclusters = atoi(argv[2]);
    char filename[2048];

    iftImage *img = iftReadImageByExt(fs->files[0]->path);
    int nfeats = img->n;
    iftDestroyImage(&img);

    iftDataSet *Z         = iftCreateDataSet(fs->n, nfeats);
    iftFileSet *fsRefData = iftCreateFileSet(fs->n);
    Z->nclasses = 1;

    /* Creating image dataset */

    for (int i = 0; i < fs->n; i++){
        printf("Processing file %d/%ld\r",i+1,fs->n);
        fflush(stdout);

        img = iftReadImageByExt(fs->files[i]->path);

        if (img->n != Z->nfeats){
            iftError("Image %s is in a different domain","main",fs->files[i]->path);
        }

        Z->sample[i].id = i;
        Z->sample[i].truelabel = 1;
        for (int f = 0; f < img->n; f++){
            Z->sample[i].feat[f] = img->val[f];
        }

        char *abs_path = iftAbsPathname(fs->files[i]->path);
        sprintf(filename, "%s", abs_path);
        fsRefData->files[i] = iftCreateFile(filename);
        free(abs_path);

        iftDestroyImage(&img);
    }
    puts("");
    iftDestroyFileSet(&fs);
    iftSetRefData(Z, fsRefData, IFT_REF_DATA_FILESET);
    iftSetStatus(Z, IFT_TRAIN);
    iftAddStatus(Z, IFT_SUPERVISED);

    /* Applying Clustering */
    iftKMeans(Z,nclusters,100,0.001,NULL,NULL);

    iftWriteDataSet(Z, argv[3]);
    printf("File '%s' created\n", argv[3]);
    iftDestroyDataSet(&Z);
    iftDestroyFileSet(&fsRefData);

    return 0;
}