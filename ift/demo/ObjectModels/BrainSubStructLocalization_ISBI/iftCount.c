#include "ift.h"


int main(int argc, const char *argv[]) {
    if (argc != 2)
        iftError("iftCount <labels.csv>", "main");

    iftFileSet *labels_set = iftLoadFileSetFromDirOrCSV(argv[1], 0, true);

    for (int f = 0; f < labels_set->n; f++) {
        iftImage *label_img = iftReadImageByExt(labels_set->files[f]->path);

        int counts[3] = {0, 0, 0};

        for (int p = 0; p < label_img->n; p++)
            counts[label_img->val[p]]++;

        puts(iftFilename(labels_set->files[f]->path, NULL));
        printf("[1] = %d\n", counts[1]);
        printf("[2] = %d\n", counts[2]);
        printf("diff = %d\n\n", counts[1] - counts[2]);


        iftDestroyImage(&label_img);
    }


    return 0;
}






