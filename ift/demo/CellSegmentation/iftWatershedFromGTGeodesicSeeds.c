#include "ift.h"

iftDict *iftGetArguments(int argc, const char *argv[]);

int main(int argc, const char *argv[])
{
    iftImageForest        *fst = NULL;
    iftImage        *img=NULL,*basins = NULL, *gt = NULL;
    iftImage        *marker=NULL;
    iftAdjRel       *A=NULL, *B=NULL;
    timer           *t1=NULL,*t2=NULL;
    iftDict         *args = NULL;
    bool            bkg_forbidden;

    /*--------------------------------------------------------*/
    size_t MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/


    args = iftGetArguments(argc, argv);

    img = iftReadImageByExt(iftGetConstStrValFromDict("--input-img", args));
    gt = iftReadImageByExt(iftGetConstStrValFromDict("--gt", args));

    bkg_forbidden = iftDictContainKey("--bkg-forbidden", args, NULL);

    iftPrintDict(args);


    /* the operation is connected for the topology defined by A: A must
       be the same in all operators (including
       iftVolumeClose?). Otherwise, this is not a connected operation in
       the desired topology. */

    if(iftIs3DImage(img)) {
        A = iftSpheric(1.0);
        B = iftSpheric(iftGetDblValFromDict("--radius", args));
    } else {
        A = iftCircular(1.5);
        B = iftCircular(iftGetDblValFromDict("--radius", args));
    }
    if(iftDictContainKey("--basins", args, NULL)) {
        basins = iftReadImageByExt(iftGetConstStrValFromDict("--basins", args));
    } else {
        basins = iftImageBasins(img, A);
    }

    if(iftIs3DImage(img))
        iftWriteImageByExt(basins,"basins.scn");
    else
        iftWriteImageByExt(basins, "basins.pgm");

    t1 = iftTic();

    fst = iftCreateImageForest(basins, A);
    // Hack to select a seed on the background
    iftImage *gt_tmp = iftAddValue(gt, 1);
    iftLabeledSet *S = iftGeodesicCenters(gt_tmp);

    iftLabeledSet *Seeds = NULL;
    for(iftLabeledSet *Saux = S; Saux != NULL; Saux = Saux->next) {
        int p = Saux->elem;
        int lb = Saux->label;

        iftVoxel u = iftGetVoxelCoord(gt_tmp, p);
        for(int i = 0; i < B->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(B, u, i);

            if(iftValidVoxel(gt_tmp, v)) {
                if(iftImgVoxelVal(gt_tmp, v) == lb) {
                    if(!bkg_forbidden || lb > 1) {
                        iftInsertLabeledSet(&Seeds, iftGetVoxelIndex(gt_tmp, v), lb - 1);
                    }
                }
            }
        }
    }

    // Preventing competition to take place in the background region of the ground truth
    if(bkg_forbidden) {
        for(int p = 0; p < fst->img->n; p++) {
            if(gt->val[p] == 0) {
                fst->pathval->val[p] = IFT_INFINITY_INT_NEG;
            }
        }
    }

    iftDiffWatershed(fst, Seeds, NULL);

    t2     = iftToc();

    fprintf(stdout,"Watershed in %f ms with %d regions\n",iftCompTime(t1,t2),iftMaximumValue(fst->label));

    if(iftIs3DImage(img))
        iftWriteImageByExt(fst->label,"result.scn");
    else
        iftWriteImageByExt(fst->label, "result.pgm");

    if(iftIs3DImage(img)) {
        fprintf(stderr, "pred min val %d, max val %d\n", iftMinimumValue(fst->pred), iftMaximumValue(fst->pred));
        iftWriteRawSceneWithInfoOnFilename(fst->pred, "pred.raw");
    } else {
        iftWriteImageP2(fst->pred, "pred.pgm");

        iftImage *seed_img = iftSeedImageFromLabeledSet(Seeds, fst->label);
        iftColorTable *cmap = iftCreateColorTable(255, RGB_CSPACE);
        iftDrawSeeds(img, seed_img, cmap);
        iftWriteImageByExt(img, "seeds.png");
        iftDrawLabels(img, fst->label, cmap, A, true, 0.5);
        iftWriteImageByExt(img, "result.png");
        iftDestroyImage(&seed_img);
        iftDestroyColorTable(&cmap);
    }
    iftWriteSeeds(Seeds, fst->label, "seeds.txt");


    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
    iftDestroyImage(&img);
    iftDestroyImage(&gt);
    iftDestroyImage(&gt_tmp);
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyImageForest(&fst);
    iftDestroyLabeledSet(&S);
    iftDestroyLabeledSet(&Seeds);
    iftDestroyDict(&args);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

    return(0);
}

iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = "This program segments an input image with watershed by selecting one seed at the geodesic centers of the ground truth labels";

    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input image"},
            {.short_name = "-b", .long_name = "--basins", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Image basins"},
            {.short_name = "-r", .long_name = "--radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Seed radius"},
            {.short_name = "-g", .long_name = "--gt", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Ground truth image"},
            {.short_name = "", .long_name = "--bkg-forbidden", .has_arg=false, .arg_type=IFT_BOOL_TYPE,
                    .required=false, .help="Forbids competition to occur in the background"}
    };
    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}
