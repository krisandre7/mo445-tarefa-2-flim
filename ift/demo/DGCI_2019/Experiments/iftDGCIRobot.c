#include "ift.h"

int main(int argc, const char *argv[])
{
    if (argc != 5)
        iftError("input: iftDGCIRobot <original dir> <ground truth dir> <output dir> <segmentation method>",
                "iftDGCIRobot");

    /* parameters */
    int iterations = 15;
    float accuracy = 0.98f;
    float marker_radius = 1.0f;
    float marker_limit = 4.0f;
    int seeds_per_ite = 1;
    iftAdjRel *A = iftCircular(1.0f);
    int gc_beta = 30;
    /* end parameters */

    const char *orig_dir   = argv[1];
    const char *gt_dir     = argv[2];
    const char *output_dir = argv[3];
    const char *method     = argv[4];

    iftFileSet *orig_set = iftLoadFileSetFromDir(orig_dir, 1);
    iftFileSet *gt_set = iftLoadFileSetFromDir(gt_dir, 1);

    if (orig_set->n != gt_set->n) {
        iftError("Ground truth images and original images directory have different"
                 "quantity of files", "iftDGCIRobot");
    }

    for (int i = 0; i < orig_set->n; i++)
    {
        char *orig_path = orig_set->files[i]->path;
        char *gt_path = gt_set->files[i]->path;
        char *aux = iftFilename(orig_path, NULL);
        char *name = iftBasename(aux);
        iftFree(aux);

        iftRobot *bot     = iftCreateRobot(orig_path, gt_path, iterations, accuracy);
        iftMImage *mimg   = iftImageToMImage(bot->orig, LABNorm_CSPACE);
        iftImage *basins  = iftMImageBasins(mimg, A);

        /* robot setup */
        iftDestroyAdjRel(&bot->mrk_radius);
        iftDestroyAdjRel(&bot->limit_mrk_rad);
        bot->mrk_radius = iftCircular(marker_radius);
        bot->limit_mrk_rad = iftCircular(marker_limit);
        bot->max_seed_per_iter = seeds_per_ite;
        bot->seeds = NULL; /* Initial seeds maybe selected */
        /* end robot setup*/

        printf("Image: %s\n", name);

        do {
            /* Check if seeds have already been computed */
            if (bot->seeds != NULL) {
                iftDestroyImage(&bot->segm);

                /* segmentation method */
                if (strcmp(method, "object") == 0) {
                    bot->segm = iftDynamicSetObjectPolicy(mimg, A, bot->seeds, false);
                } else if (strcmp(method, "root") == 0) {
                    bot->segm = iftDynamicSetRootPolicy(mimg, A, bot->seeds, 1, false);
                } else if (strcmp(method, "minroot") == 0) {
                    bot->segm = iftDynamicSetMinRootPolicy(mimg, A, bot->seeds, 1, false);
                } else if (strcmp(method, "object_dist") == 0) {
                    bot->segm = iftDynamicSetObjectPolicy(mimg, A, bot->seeds, true);
                } else if (strcmp(method, "root_dist") == 0) {
                    bot->segm = iftDynamicSetRootPolicy(mimg, A, bot->seeds, 1, true);
                } else if (strcmp(method, "minroot_dist") == 0) {
                    bot->segm = iftDynamicSetMinRootPolicy(mimg, A, bot->seeds, 1, true);
                } else if (strcmp(method, "watershed") == 0) {
                    bot->segm = iftWatershed(basins, A, bot->seeds, NULL);
                } else if (strcmp(method, "watercut") == 0) {
                    bot->segm = iftWaterCut(mimg, A, bot->seeds, NULL);
                } else if (strcmp(method, "graphcut") == 0) {
                    bot->segm = iftGraphCut(basins, bot->seeds, gc_beta);
                } /* Adicionar me'todos que utilizam mapas de objetos e superpixels */
                else {
                    iftError("Segmentation method not found", "iftDGCIRobot");
                }
                /* end segmentation method */
            }

            iftRobotUpdateError(bot);
            iftRobotPrintInfo(bot);

            iftRobotFindSeedsMSSkel(bot);
            if (bot->converged == true) {
                bot->converged = false;
                iftRobotFindSeedsCenterOfMass(bot);
            }
        } while (iftRobotContinue(bot));

        iftImage *result = iftMask(bot->orig, bot->segm);
        char *out_path = iftConcatStrings(6, output_dir, "/", method, "_", name, ".png");
        iftWriteImageByExt(result, out_path);

        iftFree(name);
        iftFree(out_path);

        iftDestroyRobot(&bot);
        iftDestroyMImage(&mimg);
        iftDestroyImage(&basins);
        iftDestroyImage(&result);
    }

//    iftDestroyFileSet(&orig_set);
//    iftDestroyFileSet(&gt_set);

    iftDestroyAdjRel(&A);

    return 0;
}
