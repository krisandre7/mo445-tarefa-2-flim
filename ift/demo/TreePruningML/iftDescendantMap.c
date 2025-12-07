//
// Created by Thiago Vallin Spina on 5/7/17.
//


#include "ift.h"

iftDict *iftGetArguments(int argc, const char *argv[]);
void iftDescendantMapsBySegmentationError(iftImage *pred, iftImage *label, iftImage *gt, iftAdjRel *A, iftBMap *set_of_interest,
                                          iftImage **descendants_correct, iftImage **descendants_error);
int main(int argc, const char *argv[]) {
    iftImage *pred = NULL, *descendants = NULL, *set_of_interest_img = NULL, *label = NULL;
    iftImage *descendants_correct = NULL, *descendants_error = NULL;
    iftAdjRel *A = NULL;
    iftDict *args = NULL;
    iftBMap *set_of_interest = NULL;
    double radius;

    /*--------------------------------------------------------*/
    size_t MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/

    args = iftGetArguments(argc, argv);

    pred    = iftReadRawSceneWithInfoOnFilename(iftGetConstStrValFromDict("--pred", args));
    label   = iftReadImageByExt(iftGetConstStrValFromDict("--label", args));
    radius  = iftGetDblValFromDict("--radius", args);

    if(iftIs3DImage(pred))
        A = iftSpheric(radius);
    else
        A = iftCircular(radius);

    if(iftDictContainKey("--set-of-interest", args, NULL)) {
        set_of_interest_img = iftReadImageByExt(iftGetConstStrValFromDict("--set-of-interest", args));
        set_of_interest = iftBinImageToBMap(set_of_interest_img);
    } else if(iftDictContainKey("--from-forest-borders", args, NULL)){
        set_of_interest_img = iftObjectBorders(label, A, false, true);
        set_of_interest = iftBinImageToBMap(set_of_interest_img);
    }

    descendants = iftDescendantMap(pred, A, set_of_interest);

    char *output_basename = iftBasename(iftGetConstStrValFromDict("--output-descendants", args));
    char *output_path = NULL;

    if(iftIs3DImage(pred))
        output_path = iftConcatStrings(2, output_basename, ".scn");
    else
        output_path = iftConcatStrings(2, output_basename, ".pgm");

    iftWriteImageByExt(descendants, output_path);
    iftFree(output_path);

    if(iftDictContainKey("--gt", args, NULL)) {
        iftImage *gt    = iftReadImageByExt(iftGetConstStrValFromDict("--gt", args));

        iftDescendantMapsBySegmentationError(pred, label, gt, A, set_of_interest,
                                             &descendants_correct, &descendants_error);

        if(iftIs3DImage(pred)) {
            output_path = iftConcatStrings(2, output_basename, "_error.scn");
        } else {
            output_path = iftConcatStrings(2, output_basename, "_error.pgm");
        }

        iftWriteImageByExt(descendants_error, output_path);

        iftFree(output_path);

        if(iftIs3DImage(pred)) {
            output_path = iftConcatStrings(2, output_basename, "_correct.scn");
        } else {
            output_path = iftConcatStrings(2, output_basename, "_correct.pgm");
        }

        iftWriteImageByExt(descendants_correct, output_path);

        iftFree(output_path);

        iftDestroyImage(&gt);
    }

    iftFree(output_basename);

    iftDestroyImage(&label);
    iftDestroyDict(&args);
    iftDestroyAdjRel(&A);
    iftDestroyImage(&pred);
    iftDestroyImage(&descendants);
    iftDestroyImage(&set_of_interest_img);
    iftDestroyBMap(&set_of_interest);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%lu, %lu)\n",
               MemDinInicial,MemDinFinal);

    return 0;
}


iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = "This is a program computes the descendant map of an optimum path forest";

    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-p", .long_name = "--pred", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Optimum Path Forest (raw image file with header on filename, see iftReadRawSceneWithInfoOnFilename)"},
            {.short_name = "-l", .long_name = "--label", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input segmentation label"},
            {.short_name = "-r", .long_name = "--radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Adjacency radius"},
            {.short_name = "-o", .long_name = "--output-descendants", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output descendant map"},
            {.short_name = "-g", .long_name = "--gt", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Input ground truth label for computing the descendants in/correctly segmented"},
            {.short_name = "-s", .long_name = "--set-of-interest", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Binary image containing the set of voxels from which the descendant map should be computed"},
            {.short_name = "", .long_name = "--from-forest-borders", .has_arg=false, .arg_type=IFT_BOOL_TYPE,
                .required=false, .help="If set, the descendant map will count only the voxels in the forest's border. Note: --set-of-interest overrides this parameter."}
    };
    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}



void iftDescendantMapsBySegmentationError(iftImage *pred, iftImage *label, iftImage *gt, iftAdjRel *A, iftBMap *set_of_interest,
                                          iftImage **descendants_correct, iftImage **descendants_error) {
    iftFIFO *Q = NULL;
    iftLIFO *S = NULL;
    bool count_on_set = (set_of_interest != NULL);

    (*descendants_correct)  = iftCreateImage(pred->xsize, pred->ysize, pred->zsize);
    (*descendants_error)    = iftCreateImage(pred->xsize, pred->ysize, pred->zsize);
    Q = iftCreateFIFO(pred->n);
    S = iftCreateLIFO(pred->n);

    for(int p = 0; p < pred->n; p++) {
        if(pred->val[p] == IFT_NIL) {
            iftInsertFIFO(Q, p);
        }
    }

    while(!iftEmptyFIFO(Q)) {
        int p = iftRemoveFIFO(Q);
        iftVoxel u = iftGetVoxelCoord(pred, p);
        iftInsertLIFO(S, p);

        for(int i = 1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if(iftValidVoxel(pred, v) && iftImgVoxelVal(pred, v) == p) {
                iftInsertFIFO(Q, iftGetVoxelIndex(pred, v));
            }
        }
    }

    while(!iftEmptyLIFO(S)) {
        int p = iftRemoveLIFO(S);
        bool correct = label->val[p] == gt->val[p];

        if(pred->val[p] != IFT_NIL) {
            // Counting all leaves/tree nodes as descendants, taking into account whether they have been correctly
            // or incorrectly labeled by segmentation
            if(!count_on_set) {
                if(correct) {
                    (*descendants_correct)->val[pred->val[p]] += (*descendants_correct)->val[p] + 1;
                    (*descendants_error)->val[pred->val[p]] += (*descendants_error)->val[p];
                } else {
                    (*descendants_correct)->val[pred->val[p]] += (*descendants_correct)->val[p];
                    (*descendants_error)->val[pred->val[p]] += (*descendants_error)->val[p] + 1;
                }
            } else {
                // Counting only (*descendants_correct) in the set of interest voxels
                (*descendants_correct)->val[pred->val[p]] += (*descendants_correct)->val[p];
                (*descendants_error)->val[pred->val[p]] += (*descendants_error)->val[p];
               if(iftBMapValue(set_of_interest, p)) {
                    if(correct) {
                        (*descendants_correct)->val[pred->val[p]]++;
                    } else {
                        (*descendants_error)->val[pred->val[p]]++;
                    }
                }
            }
        }
    }
}



