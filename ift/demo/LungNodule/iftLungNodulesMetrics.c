//
// Created by azaelmsousa on 29/04/21.
//

#include "ift.h"

int main(int argc, char *argv[])
{
    if (argc != 4){
        iftError("Usage: iftLungNodulesMetrics <...>\n[1] Input directory with resulting annotations\n"
                 "[2] Input directory with annotation files\n"
                 "[3] Output csv file with similarity results","main");
    }

    char *segm_dir = argv[1];
    char *ann_dir = argv[2];
    char *out_path = argv[3];

    iftFileSet *segm_fs = iftLoadFileSetFromDirBySuffix(segm_dir,".nii.gz",0);
    iftFileSet *ann_fs = iftLoadFileSetFromDirBySuffix(ann_dir,".nii.gz",0);

    if (segm_fs->n != ann_fs->n){
        iftError("Segmentation and annotation dirs have different number of files","main");
    }

    for (int i = 0; i < segm_fs->n; i++){
        iftImage *segm = iftReadImageByExt(segm_fs->files[i]->path);
        iftImage *ann = iftReadImageByExt(ann_fs->files[i]->path);
        char *segm_filename = iftFilename(segm_fs->files[i]->path,NULL);
        char *ann_filename = iftFilename(ann_fs->files[i]->path,NULL);

        iftImage *segm_ths = iftThreshold(segm,1,iftMaximumValue(segm),1);
        iftDestroyImage(&segm_ths);
        iftImage *ann_ths = iftThreshold(ann,1,iftMaximumValue(ann),1);
        iftDestroyImage(&ann_ths);

        double dice = iftDiceSimilarity(segm_ths,ann_ths);
        iftDestroyImage(&segm_ths);
        iftDestroyImage(&ann_ths);
        printf("%s/%s >> %.2f\n",segm_filename,ann_filename,dice);

        // Writing output file
        FILE *f;
        bool firstline = FALSE;
        if (!iftFileExists(out_path))
            firstline = TRUE;
        f = fopen(out_path, "a+");
        if (f == NULL)
            iftError("Cannot open csv file", "main");
        if (firstline)
            fprintf(f, "patient;dice\n");
        fprintf(f, "%s;%lf\n",
                segm_filename, //patient
                dice);
        fclose(f);
        free(segm_filename);
        free(ann_filename);
    }

    return 0;
}