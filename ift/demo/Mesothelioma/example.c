#include <ift.h>

int main(int argc, char *argv[]) {
    if (argc != 4)
        iftError("input: iftRobot <original dir> <ground truth dir> <output dir>",
                 "iftRobot");

    const char *orig_dir   = argv[1];
    const char *gt_dir     = argv[2];
    const char *out_dir    = argv[3];

                                                                                           

    iftMakeDir(out_dir);

    if (orig_set->n != gt_set->n) {
        iftError("Ground truth images and original images directory have different"
                 "quantity of files", "iftRobot");
    }
 
    iftImage **imgs = (iftImage**) iftAlloc(orig_set->n, sizeof(iftImage*));

    for (int i = 0; i < orig_set->n; i++) {
        char *orig_path = orig_set->files[i]->path;
        char *gt_path   = gt_set->files[i]->path;
        char *filename  = iftFilename(orig_path, NULL);

        imgs[i] = iftReadImageByExt(orig_path);

        iftWriteSeeds(orig, "%s/%s-seeds.txt", out_dir, filename);
        iftWriteImageByExt(orig, "%s/%s", out_dir, filename);

        iftDestroyImage(&orig);
    }


    iftDestroyFileSet(&orig_set);
    iftDestroyFileSet(&gt_set);

    return 0;
}

