#include "ift.h"

iftDict *iftGetArguments(int argc, const char *argv[]);
void iftWaterGrayForestFromGT(iftImageForest *fst, iftImage *marker, iftImage *gt);

int main(int argc, const char *argv[])
{
    iftImageForest        *fst = NULL;
    iftImage        *img=NULL,*basins = NULL, *gt = NULL;
    iftImage        *marker=NULL;
    iftAdjRel       *A=NULL, *B=NULL;
    timer           *t1=NULL,*t2=NULL;
    iftDict         *args = NULL;
    bool bkg_forbidden;

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
    } else {
        A = iftCircular(1.5);
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

    if(iftDictContainKey("--height", args, NULL)) {
        marker = iftAddValue(basins, (int) iftGetLongValFromDict("--height", args));
    } else if(iftDictContainKey("--volume", args, NULL)) {
        marker = iftVolumeClose(basins, (int) iftGetLongValFromDict("--volume", args));
    } else {
        iftError("Please provide either the --height or the --volume parameter to create the grayscale markers", "main");
    }

    // Preventing competition to take place in the background region of the ground truth
    if(bkg_forbidden) {
        for(int p = 0; p < fst->img->n; p++) {
            if(gt->val[p] == 0) {
                marker->val[p] = IFT_INFINITY_INT_NEG;
            }
        }
    }

    iftWaterGrayForestFromGT(fst, marker, gt);

    t2     = iftToc();

    fprintf(stdout,"WaterGray in %f ms with %d regions\n",iftCompTime(t1,t2),iftMaximumValue(fst->label));


    if(iftIs3DImage(img))
        iftWriteImageByExt(fst->label,"result.scn");
    else
        iftWriteImageByExt(fst->label, "result.pgm");

    iftLabeledSet *Seeds = NULL;

    iftIntArray *nseeds_per_label = iftCreateIntArray(iftMaximumValue(gt)+1);

    // Creating the seed set from the predecessor map
    for(int p = 0; p < fst->pred->n; p++) {
        if(fst->pred->val[p] < 0) {
            if(!bkg_forbidden || gt->val[p] > 0) {
                iftInsertLabeledSet(&Seeds, p, fst->label->val[p]);
            }

            nseeds_per_label->val[fst->label->val[p]]++;
        }
    }

    int nlabels_with_two_seeds = 0;
    int nlabels_without_seeds = 0;
    for(int lb = 1; lb < nseeds_per_label->n; lb++) {
        if(nseeds_per_label->val[lb] > 1) {
            nlabels_with_two_seeds++;
        } else if(nseeds_per_label->val[lb] == 0) {
            nlabels_without_seeds++;
        }
    }

    fprintf(stdout, "Number of labels with two seeds: %03d/%03d\nNumber of labels without seeds: %03d/%03d\n",
            nlabels_with_two_seeds, nseeds_per_label->n, nlabels_without_seeds, nseeds_per_label->n);
    iftImage *two_seeds = iftCopyImage(fst->label);
    iftImage *no_seeds  = iftCopyImage(gt);

    for(int p = 0; p < fst->label->n; p++) {
        if(fst->label->val[p] > 0) {
            if(nseeds_per_label->val[fst->label->val[p]] <= 1) {
                two_seeds->val[p] = 0;
            }
        }

        if(gt->val[p] > 0) {
            if(nseeds_per_label->val[gt->val[p]] >= 1) {
                no_seeds->val[p] = 0;
            }
        }
    }

    if(iftIs3DImage(img)) {
        iftWriteRawSceneWithInfoOnFilename(fst->pred, "pred.raw");

        iftWriteImageByExt(two_seeds, "two_seeds.scn");
        iftWriteImageByExt(no_seeds, "no_seeds.scn");

    } else {
        iftWriteImageP2(fst->pred, "pred.pgm");


        iftImage *seed_img = iftSeedImageFromLabeledSet(Seeds, fst->label);
        iftColorTable *cmap = iftCreateColorTable(iftMaximumValue(gt)+1, RGB_CSPACE);
        iftDrawSeeds(img, seed_img, cmap);
        iftWriteImageByExt(img, "seeds.png");
        iftDrawLabels(img, fst->label, cmap, A, true, 0.5);
        iftWriteImageByExt(img, "result.png");
        iftDestroyImage(&seed_img);
        iftDestroyColorTable(&cmap);

        iftWriteImageByExt(two_seeds, "two_seeds.pgm");
        iftWriteImageByExt(no_seeds, "no_seeds.pgm");
    }
    iftWriteSeeds(Seeds, fst->label, "seeds.txt");


    iftDestroyImage(&two_seeds);
    iftDestroyImage(&no_seeds);

    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
    iftDestroyImage(&img);
    iftDestroyImage(&gt);
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyImageForest(&fst);
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
    char program_description[2048] = "This program segments an input image with WaterGray by selecting the label of seeds from the ground truth";

    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input image"},
            {.short_name = "-b", .long_name = "--basins", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Image basins"},
            {.short_name = "-g", .long_name = "--gt", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Ground truth image"},
            {.short_name = "", .long_name = "--height", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="Height used to implicitly compute the H-minima transform"},
            {.short_name = "", .long_name = "--volume", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="Value used to close basins by volume"},
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


void iftWaterGrayForestFromGT(iftImageForest *fst, iftImage *marker, iftImage *gt)
{
    iftImage   *label=fst->label, *basins=fst->img, *pathval=fst->pathval;
    iftImage   *pred=fst->pred, *root=fst->root;
    iftGQueue  *Q=fst->Q;
    iftAdjRel  *A=fst->A;
    int         i,p,q,tmp;
    iftVoxel    u,v;

    // Initialization


    for (p=0; p < pathval->n; p++) {
        pathval->val[p] = marker->val[p] + 1;
        iftInsertGQueue(&Q,p);
    }

    // Image Foresting Transform

    while(!iftEmptyGQueue(Q)) {
        p=iftRemoveGQueue(Q);

        if (pred->val[p] == IFT_NIL) { // root voxel
            pathval->val[p]  -= 1;
            label->val[p]=gt->val[p];
        }

        u = iftGetVoxelCoord(basins,p);

        for (i=1; i < A->n; i++){
            v = iftGetAdjacentVoxel(A,u,i);
            if (iftValidVoxel(basins,v)){
                q = iftGetVoxelIndex(basins,v);
                if (pathval->val[q] > pathval->val[p]){
                    tmp = iftMax(pathval->val[p], basins->val[q]);
                    if (tmp < pathval->val[q]){
                        iftRemoveGQueueElem(Q,q);
                        label->val[q]      = label->val[p];
                        root->val[q]       = root->val[p];
                        pred->val[q]       = p;
                        pathval->val[q]    = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

}
