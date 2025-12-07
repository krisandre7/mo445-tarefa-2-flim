//
// Created by azaelmsousa on 19/03/21.
//

#include "ift.h"

int main(int argc, char *argv[])
{
    if (argc != 4){
        iftError("Usage iftAlignLungsByPCA <...>\n[1] Input dir with asymmetrical images\n[2] Dir with label images\n[3] Output dir","main");
    }

    iftFileSet *fs_orig  = iftLoadFileSetFromDirOrCSV(argv[1],1,1);
    iftFileSet *fs_label = iftLoadFileSetFromDirOrCSV(argv[2],1,1);

    if (fs_orig->n != fs_label->n)
        iftError("Image dir and Label dir have different sizes","main");

    for (int i = 0; i < fs_orig->n; i++){
        char *labelname = iftFilename(fs_label->files[i]->path,NULL);
        char *imgname = iftFilename(fs_orig->files[i]->path,NULL);
        printf("Processing file %d/%ld\r",i+1,fs_orig->n);
        fflush(stdout);

        iftImage *lbl = iftReadImageByExt(fs_label->files[i]->path);
        iftMatrix *Mrot = iftRotationMatrixToAlignByPrincipalAxes(lbl);
        iftDestroyImage(&lbl);

        iftImage *img = iftReadImageByExt(fs_orig->files[i]->path);
        iftImage *aligned = iftTransformImageByMatrix(img,Mrot);
        aligned->dx = 1.;
        aligned->dy = 1.;
        aligned->dz = 1.;
        iftDestroyImage(&img);
        iftDestroyMatrix(&Mrot);

        char outname[200];
        sprintf(outname,"%s/%s",argv[3],imgname);
        iftWriteImageByExt(aligned,outname);
        iftDestroyImage(&aligned);
        free(labelname);
        free(imgname);
    }
    iftDestroyFileSet(&fs_orig);
    iftDestroyFileSet(&fs_label);

    return 0;
}