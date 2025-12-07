//
// Created by Thiago Vallin Spina on 5/7/17.
//

#include "ift.h"

#define IFT_LEAKING_CLASS 1
#define IFT_BORDER_CLASS 2
#define IFT_INTERIOR_CORRECT_CLASS 3
#define IFT_INTERIOR_ERROR_CLASS 4


iftDict *iftGetArguments(int argc, const char *argv[]);
iftDataSet *iftDescendantMapToDataSet(iftImage *descendants, iftImage *label, iftImage *gt, iftAdjRel *A, bool use_inner_nodes);
iftBMap *iftSelectLeakingPointsForTraining(iftImage *descendants, iftLabeledSet *seeds, iftImage *pred, iftAdjRel *A,
                                           iftImage *label, iftImage *gt, double leaking_min_cutoff,
                                           double leaking_max_cutoff, iftImage *descendants_correct,
                                           iftImage *descendants_error);

int main(int argc, const char *argv[]) {
    iftAdjRel *A = NULL, *B = NULL;
    iftImage *descendants = NULL, *label = NULL, *gt = NULL, *pred = NULL;
    iftImage *descendants_correct = NULL, *descendants_error = NULL;
    iftDataSet *Z = NULL;
    iftLabeledSet *seeds = NULL;

    iftDict *args = NULL;
    double radius;
    const char *output_dataset = NULL;
    bool use_inner_nodes = false;

    /*--------------------------------------------------------*/
    size_t MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/

    args = iftGetArguments(argc, argv);

    descendants = iftReadImageByExt(iftGetConstStrValFromDict("--descendants", args));

    char *bname = iftBasename(iftGetConstStrValFromDict("--descendants", args));
    char *path  = iftConcatStrings(3, bname,  "_correct", iftFileExt(iftGetConstStrValFromDict("--descendants", args)));
    if(iftFileExists(path)) {
        descendants_correct = iftReadImageByExt(path);
    }
    iftFree(path);

    path        = iftConcatStrings(3, bname,  "_error", iftFileExt(iftGetConstStrValFromDict("--descendants", args)));
    if(iftFileExists(path)) {
        descendants_error = iftReadImageByExt(path);
    }

    iftFree(path);

    pred            = iftReadRawSceneWithInfoOnFilename(iftGetConstStrValFromDict("--pred", args));
    label           = iftReadImageByExt(iftGetConstStrValFromDict("--label", args));
    gt              = iftReadImageByExt(iftGetConstStrValFromDict("--gt", args));
    radius          = iftGetDblValFromDict("--radius", args);
    use_inner_nodes = iftDictContainKey("--use-inner-nodes", args, NULL);
    output_dataset  = iftGetConstStrValFromDict("--output-dataset", args);
    seeds           = iftReadSeeds(pred, iftGetConstStrValFromDict("--seeds", args));

    fprintf(stderr,"Max val %d\n", iftMaximumValue(descendants));
    if(iftIs3DImage(descendants)) {
        A = iftSpheric(radius);
        B = iftSpheric(1.0);
    } else {
        A = iftCircular(radius);
        B = iftCircular(1.5);
    }
    iftBMap *leakings = iftSelectLeakingPointsForTraining(descendants, seeds, pred, B, label, gt,
                                                          iftGetDblValFromDict("--min-perc", args),
                                                          iftGetDblValFromDict("--max-perc", args),
                                                          descendants_correct, descendants_error);

    iftDestroyBMap(&leakings);
//    Z = iftDescendantMapToDataSet(descendants, label, gt, A, use_inner_nodes);
//
//    if(iftDictContainKey("--normalize-dataset", args, NULL)) {
//        iftDataSet *Zun = iftUnitNormDataSet(Z);
//        iftDestroyDataSet(&Z);
//        Z = Zun;
//    }
//
//    iftWriteOPFDataSet(Z, output_dataset);

//    /** Visualizaing the data set **/
//    fprintf(stderr,"Here\n");
//    iftSetStatus(Z, IFT_TEST);
//    iftDataSet* Zvis = iftDimReductionByTSNE(Z, 2, 40, 1000, false);
//    fprintf(stderr,"Here1\n");
//
//    iftImage *outputImage = iftDraw2DFeatureSpace(Zvis,CLASS,IFT_TEST);
//    iftWriteImageP6(outputImage,"t_sne.ppm");
//    iftDestroyDataSet(&Zvis);
//    iftDestroyImage(&outputImage);

    iftDestroyAdjRel(&A);
    iftDestroyImage(&label);
    iftDestroyImage(&descendants);
    iftDestroyImage(&pred);
    iftDestroyDict(&args);
    iftDestroyDataSet(&Z);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%lu, %lu)\n",
               MemDinInicial,MemDinFinal);

    return 0;
}



iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = "This is a program that converts a descendant map into a data set for machine learning";

    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-d", .long_name = "--descendants", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Descendant map of an Optimum Path Forest"},
            {.short_name = "-l", .long_name = "--label", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input segmentation label"},
            {.short_name = "-p", .long_name = "--pred", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Optimum Path Forest (raw image file with header on filename)"},
            {.short_name = "-g", .long_name = "--gt", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Ground truth segmentation mask"},
            {.short_name = "-r", .long_name = "--radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Adjacency radius used to normalize and compute the feature vector"},
            {.short_name = "-o", .long_name = "--output-dataset", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Output data set"},
            {.short_name = "", .long_name = "--seeds", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Seeds corresponding to the roots of the forest"},
            {.short_name = "", .long_name = "--min-perc", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Leaking size min percentage threshold"},
            {.short_name = "", .long_name = "--max-perc", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Leaking size max percentage threshold"},

            {.short_name = "", .long_name = "--use-inner-nodes", .has_arg=false, .arg_type=IFT_BOOL_TYPE,
                .required=false, .help="If set, the inner nodes will be considered as another class"},
            {.short_name = "", .long_name = "--normalize-dataset", .has_arg=false, .arg_type=IFT_BOOL_TYPE,
                .required=false, .help="If set, every feature vector will be normalized"}
    };
    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

iftBMap *iftSelectLeakingPointsForTraining(iftImage *descendants, iftLabeledSet *seeds, iftImage *pred, iftAdjRel *A,
                                           iftImage *label, iftImage *gt, double leaking_min_cutoff,
                                           double leaking_max_cutoff, iftImage *descendants_correct,
                                           iftImage *descendants_error) {
    iftBMap *leaking_point_candidates = NULL;
    iftBMap *selected_leaking_points = NULL;
    iftFIFO *Q = NULL;
    iftSet *leaking_candidates = NULL;

    Q = iftCreateFIFO(descendants->n);

    for(iftLabeledSet *S = seeds; S != NULL; S = S->next) {
        iftInsertFIFO(Q, S->elem);
    }

    leaking_point_candidates = iftCreateBMap(pred->n);
    selected_leaking_points = iftCreateBMap(pred->n);

    while(!iftEmptyFIFO(Q)) {
        int p = iftRemoveFIFO(Q);
        iftVoxel u = iftGetVoxelCoord(pred, p);

        if(pred->val[p] != IFT_NIL) {
            if(!iftBMapValue(leaking_point_candidates, pred->val[p]) &&
                label->val[p] != gt->val[p] &&
                label->val[pred->val[p]] == gt->val[pred->val[p]]) {
                iftBMapSet1(leaking_point_candidates, pred->val[p]);
                iftInsertSet(&leaking_candidates, pred->val[p]);

                // We do not need to add the successors of p to the Q since p was used
                // to detect a leaking point
                continue;
            }
        }

        for(int i = 1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if(iftValidVoxel(pred, v) && iftImgVoxelVal(pred, v) == p) {
                int q = iftGetVoxelIndex(pred, v);

                if(Q->color[q] == IFT_WHITE) {
                    iftInsertFIFO(Q, q);
                }
            }
        }
    }

    iftDestroyFIFO(&Q);

    iftImage *border_gt = NULL;
    iftAdjRel *B = NULL;
    size_t nleakings = (size_t)iftSetSize(leaking_candidates);
    iftIntArray *leakings = NULL, *leakings_size = NULL;
    iftFloatArray *leakings_percentage = NULL;
    int max_gt_lb = iftMaximumValue(gt);
    iftIntArray *leaking_labels = NULL;
    iftIntArray *leaking_labels_max_leaking = NULL;
    iftIntArray *leaking_labels_max_leaking_size = NULL;

    if (iftIs3DImage(descendants))
        B = iftSpheric(1.0);
    else
        B = iftCircular(1.5);

    border_gt = iftObjectBorders(gt, B, false, true);


    iftHist *gt_area = iftGrayHistogramInRegion(gt, border_gt, max_gt_lb + 1, false);

//    iftHist *gt_area = iftCalcGrayImageHist(gt, max_gt_lb + 1, false);

    leakings        = iftCreateIntArray(nleakings);
    leakings_size   = iftCreateIntArray(nleakings);
    leakings_percentage = iftCreateFloatArray(nleakings);

    iftSet *S = leaking_candidates;
    for (int i = 0; S != NULL; S = S->next, i++) {
        int p = S->elem;
        leakings->val[i]        = p;
        leakings_size->val[i]   = descendants->val[p];
    }

    fprintf(stderr, "Number of leakings %lu/%d\n", nleakings, descendants->n);
    int nunsuitable_leakings = 0;
    // Determining the percentage that the leaking tree of each leaking pixel has w.r.t. the ground truth label
    for(size_t i = 0; i < leakings->n; i++) {
        int p = leakings->val[i];
        leakings_percentage->val[i] = (float)(descendants->val[p] / (float)gt_area->val[label->val[p]]);

        if(descendants_error != NULL && descendants->val[p] != descendants_error->val[p] &&
                descendants_correct->val[p] > descendants_error->val[p]
                && leakings_percentage->val[i] >= leaking_min_cutoff) {
            fprintf(stderr,
                    "Leaking %10d: descendants error difference of %5d (descendant %5d, error %5d correct %5d). Percentage %f\n",
                    p,
                    descendants->val[p] - descendants_error->val[p],
                    descendants->val[p], descendants_error->val[p],
                    descendants_correct->val[p],
                    leakings_percentage->val[i] * 100.0);
            nunsuitable_leakings++;
        }
    }

    fprintf(stderr, "Number of unsuitable leakings %d\n", nunsuitable_leakings);

    int ncandidate_leakings = 0;

    for(size_t i = 0; i < leakings->n; i++) {
        int p = leakings->val[i];

        if(descendants_error != NULL && descendants->val[p] != descendants_error->val[p] &&
           descendants_correct->val[p] <= descendants_error->val[p]
           && leakings_percentage->val[i] >= leaking_min_cutoff && leakings_percentage->val[i] <= leaking_max_cutoff) {
            fprintf(stderr,
                    "Leaking %10d: descendant error difference of %5d (descendant %5d, error %5d correct %5d). Percentage %.1f\n",
                    p,
                    descendants->val[p] - descendants_error->val[p],
                    descendants->val[p], descendants_error->val[p],
                    descendants_correct->val[p],
                    leakings_percentage->val[i] * 100.0);
            ncandidate_leakings++;
        }
    }
    fprintf(stderr, "Number of candidate leakings %d\n", ncandidate_leakings);


    iftHist *hist_of_leakings = iftCreateHist(100);

    for(size_t i = 0; i < leakings_percentage->n; i++) {
        // Leakings with size greater than the ground truth label will have a percentage of leaking greater than 1.0.
        // Hence, they are counted in the last bin
        hist_of_leakings->val[iftMin(hist_of_leakings->nbins-1, (int)(leakings_percentage->val[i] * 100.0))]++;
    }

    for(int b = 0; b < hist_of_leakings->nbins; b++) {
        hist_of_leakings->val[b] /= leakings_percentage->n;
        fprintf(stderr, "Leaking %3d %lf\n", b, hist_of_leakings->val[b]);
    }

    iftHist *acc_hist = iftCalcAccHist(hist_of_leakings);

    fprintf(stderr, "Number of expressive leakings %d\n", (int)(leakings_percentage->n*(1.0-acc_hist->val[iftMax(0,(int)(leaking_min_cutoff*100)-1)])));

    leaking_labels = iftCreateIntArray(iftMaximumValue(label)+1);
    leaking_labels_max_leaking = iftCreateIntArray(iftMaximumValue(label)+1);
    leaking_labels_max_leaking_size = iftCreateIntArray(iftMaximumValue(label)+1);

    selected_leaking_points = iftCreateBMap(descendants->n);

    for(int i = 0; i < leakings_percentage->n; i++) {
        if(leakings_percentage->val[i] >= leaking_min_cutoff && leakings_percentage->val[i] <= leaking_max_cutoff) {
            iftBMapSet1(selected_leaking_points, leakings->val[i]);

            leaking_labels->val[label->val[leakings->val[i]]] = 1;

            if(leaking_labels_max_leaking_size->val[label->val[leakings->val[i]]] < leakings_size->val[i]) {
                leaking_labels_max_leaking_size->val[label->val[leakings->val[i]]] = leakings_size->val[i];
                leaking_labels_max_leaking->val[label->val[leakings->val[i]]] = i;
            }
        }
    }

    for(int lb = 0; lb < leaking_labels_max_leaking->n; lb++) {
        if(leaking_labels_max_leaking_size->val[lb] > 0) {
            fprintf(stderr, "Max leaking for label %03d: %5d, percentage %f, (pixel %8d)\n", lb,
                    leaking_labels_max_leaking_size->val[lb],
                    leakings_percentage->val[leaking_labels_max_leaking->val[lb]]*100,
                    leakings->val[leaking_labels_max_leaking->val[lb]]);
        }
    }

    iftImage *label_filtered = iftCopyImage(label);
    for(int p = 0; p < label_filtered->n; p++) {
        label_filtered->val[p] = (leaking_labels->val[label_filtered->val[p]]) ? label_filtered->val[p] : 0;

        if(iftBMapValue(selected_leaking_points, p))
            label_filtered->val[p] = max_gt_lb*2;
    }

    if(iftIs3DImage(label)) {
        iftWriteImageByExt(label_filtered, "label_filtered.scn");
    } else {
        iftWriteImageByExt(label_filtered, "label_filtered.pgm");
    }

    iftDestroyAdjRel(&B);
    iftDestroyImage(&border_gt);
    iftDestroyIntArray(&leakings);
    iftDestroyIntArray(&leakings_size);
    iftDestroyFloatArray(&leakings_percentage);
    iftDestroyHist(&gt_area);
    iftDestroyHist(&acc_hist);
    iftDestroyHist(&hist_of_leakings);

    return selected_leaking_points;
}


//iftBMap * iftSelectLeakingPointsForTraining(iftImage *descendants, iftImage *label, iftImage *gt, double leaking_cutoff,
//                                            iftImage *descendants_correct, iftImage *descendants_error) {
//    iftImage *border_gt = NULL;
//    iftAdjRel *B = NULL;
//    size_t nleakings = 0;
//    iftIntArray *leakings = NULL, *leakings_size = NULL;
//    iftFloatArray *leakings_percentage = NULL;
//    int max_gt_lb = iftMaximumValue(gt);
//    iftHist *gt_area = iftCalcGrayImageHist(gt, max_gt_lb + 1, false);
//    iftIntArray *leaking_labels = NULL;
//    iftIntArray *leaking_labels_max_leaking = NULL;
//    iftIntArray *leaking_labels_max_leaking_size = NULL;
//    iftBMap *selected_leaking_points = NULL;
//
//
//    if (iftIs3DImage(descendants))
//        B = iftSpheric(1.0);
//    else
//        B = iftCircular(1.5);
//
//    border_gt = iftObjectBorders(gt, B, false, true);
//
//    for (int p = 0; p < border_gt->n; p++) {
//        if(border_gt->val[p] != 0) {
//            if (label->val[p] != gt->val[p]) {
//                nleakings++;
//            }
//        }
//    }
//
//    leakings        = iftCreateIntArray(nleakings);
//    leakings_size   = iftCreateIntArray(nleakings);
//    leakings_percentage = iftCreateFloatArray(nleakings);
//
//    for (int i = 0, p = 0; p < border_gt->n; p++) {
//        if(border_gt->val[p] != 0) {
//            if (label->val[p] != gt->val[p]) {
//                leakings->val[i]        = p;
//                leakings_size->val[i]   = descendants->val[p];
//
//                i++;
//            }
//        }
//    }
//
//    fprintf(stderr, "Number of leakings %lu/%d\n", nleakings, descendants->n);
//    int nunsuitable_leakings = 0;
//    // Determining the percentage that the leaking tree of each leaking pixel has w.r.t. the ground truth label
//    for(size_t i = 0; i < leakings->n; i++) {
//        int p = leakings->val[i];
//        leakings_percentage->val[i] = (float)(descendants->val[p] / (float)gt_area->val[label->val[p]]);
//
//        if(descendants_error != NULL && descendants->val[p] != descendants_error->val[p] &&
//                descendants_correct->val[p] > descendants_error->val[p]
//                && leakings_percentage->val[i] >= leaking_cutoff) {
//            fprintf(stderr,
//                    "Leaking %07d: descendants error difference of %5d (descendant %5d, error %5d correct %5d). Percentage %f\n",
//                    p,
//                    descendants->val[p] - descendants_error->val[p],
//                    descendants->val[p], descendants_error->val[p],
//                    descendants_correct->val[p],
//                    leakings_percentage->val[i] * 100.0);
//            nunsuitable_leakings++;
//        }
//    }
//
//    fprintf(stderr, "Number of unsuitable leakings %d\n", nunsuitable_leakings);
//
//    int ncandidate_leakings = 0;
//
//    for(size_t i = 0; i < leakings->n; i++) {
//        int p = leakings->val[i];
//
//        if(descendants_error != NULL && descendants->val[p] != descendants_error->val[p] &&
//           descendants_correct->val[p] <= descendants_error->val[p]
//           && leakings_percentage->val[i] >= leaking_cutoff) {
//            fprintf(stderr,
//                    "Leaking %07d: descendant error difference of %5d (descendant %5d, error %5d correct %5d). Percentage %f\n",
//                    p,
//                    descendants->val[p] - descendants_error->val[p],
//                    descendants->val[p], descendants_error->val[p],
//                    descendants_correct->val[p],
//                    leakings_percentage->val[i] * 100.0);
//            ncandidate_leakings++;
//        }
//    }
//    fprintf(stderr, "Number of candidate leakings %d\n", ncandidate_leakings);
//
//
//    iftHist *hist_of_leakings = iftCreateHist(100);
//
//    for(size_t i = 0; i < leakings_percentage->n; i++) {
//        // Leakings with size greater than the ground truth label will have a percentage of leaking greater than 1.0.
//        // Hence, they are counted in the last bin
//        hist_of_leakings->val[iftMin(hist_of_leakings->nbins-1, (int)(leakings_percentage->val[i] * 100.0))]++;
//    }
//
//    for(int b = 0; b < hist_of_leakings->nbins; b++) {
//        hist_of_leakings->val[b] /= leakings_percentage->n;
//        fprintf(stderr, "Leaking %3d %lf\n", b, hist_of_leakings->val[b]);
//    }
//
//    iftHist *acc_hist =    iftCalcAccHist(hist_of_leakings);
//
//    fprintf(stderr, "Number of expressive leakings %d\n", (int)(leakings_percentage->n*(1.0-acc_hist->val[iftMax(0,(int)(leaking_cutoff*100)-1)])));
//
//    leaking_labels = iftCreateIntArray(iftMaximumValue(label)+1);
//    leaking_labels_max_leaking = iftCreateIntArray(iftMaximumValue(label)+1);
//    leaking_labels_max_leaking_size = iftCreateIntArray(iftMaximumValue(label)+1);
//
//    selected_leaking_points = iftCreateBMap(descendants->n);
//
//    for(int i = 0; i < leakings_percentage->n; i++) {
//        if(leakings_percentage->val[i] >= leaking_cutoff) {
//            iftBMapSet1(selected_leaking_points, leakings->val[i]);
//
//            leaking_labels->val[label->val[leakings->val[i]]] = 1;
//
//            if(leaking_labels_max_leaking_size->val[label->val[leakings->val[i]]] < leakings_size->val[i]) {
//                leaking_labels_max_leaking_size->val[label->val[leakings->val[i]]] = leakings_size->val[i];
//                leaking_labels_max_leaking->val[label->val[leakings->val[i]]] = i;
//            }
//        }
//    }
//
//    for(int lb = 0; lb < leaking_labels_max_leaking->n; lb++) {
//        if(leaking_labels_max_leaking_size->val[lb] > 0) {
//            fprintf(stderr, "Max leaking for label %03d: %5d, percentage %f, (pixel %8d)\n", lb,
//                    leaking_labels_max_leaking_size->val[lb],
//                    leakings_percentage->val[leaking_labels_max_leaking->val[lb]]*100,
//                    leakings->val[leaking_labels_max_leaking->val[lb]]);
//        }
//    }
//
//    iftImage *label_filtered = iftCopyImage(label);
//    for(int p = 0; p < label_filtered->n; p++) {
//        label_filtered->val[p] = (leaking_labels->val[label_filtered->val[p]]) ? label_filtered->val[p] : 0;
//
//        if(iftBMapValue(selected_leaking_points, p))
//            label_filtered->val[p] = max_gt_lb*2;
//    }
//
//    if(iftIs3DImage(label)) {
//        iftWriteImageByExt(label_filtered, "label_filtered.scn");
//    } else {
//        iftWriteImageByExt(label_filtered, "label_filtered.pgm");
//    }
//
//    iftDestroyAdjRel(&B);
//    iftDestroyImage(&border_gt);
//    iftDestroyIntArray(&leakings);
//    iftDestroyIntArray(&leakings_size);
//    iftDestroyFloatArray(&leakings_percentage);
//    iftDestroyHist(&gt_area);
//    iftDestroyHist(&acc_hist);
//    iftDestroyHist(&hist_of_leakings);
//
//    return selected_leaking_points;
//}

iftDataSet *iftDescendantMapToDataSet(iftImage *descendants, iftImage *label, iftImage *gt, iftAdjRel *A, bool use_inner_nodes) {
    iftImage *border_gt = NULL;
    iftDataSet *Z = NULL;
    iftAdjRel *B = NULL;
    int nnodes = 0;
    int nfeats = A->n;
    int nclasses = 0;

    if(iftIs3DImage(descendants))
        B = iftSpheric(1.0);
    else
        B = iftCircular(1.5);

    border_gt = iftObjectBorders(gt, B, false, true);

    if(use_inner_nodes) {
        nnodes = descendants->n;
    } else {
        for (int p = 0; p < border_gt->n; p++) {
            if (border_gt->val[p] != 0) {
                nnodes++;
            }
        }
    }

    Z = iftCreateDataSet(nnodes, nfeats);

    int truelabels[] = {0,0,0,0};

    for(int i = 0, p = 0; p < border_gt->n; p++) {
        bool inner_node = border_gt->val[p] == 0;

        if(use_inner_nodes || !inner_node) {
            iftVoxel u = iftGetVoxelCoord(border_gt, p);

            for(int j = 0; j < A->n; j++) {
                iftVoxel v = iftGetAdjacentVoxel(A, u, j);

                if(iftValidVoxel(border_gt, v)) {
                    Z->sample[i].feat[j] = iftImgVoxelVal(descendants, v);
                }
            }

            if(!inner_node) {
                if(label->val[p] != gt->val[p]) {
                    Z->sample[i].truelabel = IFT_LEAKING_CLASS;
                } else {
                    if(iftMaxFloatArray(Z->sample[i].feat, Z->nfeats) > 0)
                        iftPrintFloatArray(Z->sample[i].feat, Z->nfeats);

                    Z->sample[i].truelabel = IFT_BORDER_CLASS;
                }
            } else if(use_inner_nodes) {
                if(label->val[p] != gt->val[p]) {
                    Z->sample[i].truelabel = IFT_INTERIOR_ERROR_CLASS;
                } else {
                    Z->sample[i].truelabel = IFT_INTERIOR_CORRECT_CLASS;
                }
            }
            truelabels[Z->sample[i].truelabel-1]++;

            nclasses = iftMax(nclasses, Z->sample[i].truelabel);

            i++;
        }
    }

    Z->nclasses = nclasses;

    iftPrintIntArray(truelabels, 4);

    iftDestroyImage(&border_gt);
    iftDestroyAdjRel(&B);

    return Z;
}

iftDataSet *iftDescendantMapToDataSetHistogramFeats(iftImage *descendants, iftImage *label, iftImage *gt, iftAdjRel *A,
                                                    bool use_inner_nodes) {
    iftImage *border_gt = NULL;
    iftDataSet *Z = NULL;
    iftAdjRel *B = NULL;
    int nnodes = 0;
    int nfeats = A->n;
    int nclasses = 0;

    if(iftIs3DImage(descendants))
        B = iftSpheric(1.0);
    else
        B = iftCircular(1.5);

    border_gt = iftObjectBorders(gt, B, false, true);

    if(use_inner_nodes) {
        nnodes = descendants->n;
    } else {
        for (int p = 0; p < border_gt->n; p++) {
            if (border_gt->val[p] != 0) {
                nnodes++;
            }
        }
    }

    Z = iftCreateDataSet(nnodes, nfeats);

    int truelabels[] = {0,0,0,0};

    for(int i = 0, p = 0; p < border_gt->n; p++) {
        bool inner_node = border_gt->val[p] == 0;

        if(use_inner_nodes || !inner_node) {
            iftVoxel u = iftGetVoxelCoord(border_gt, p);

            for(int j = 0; j < A->n; j++) {
                iftVoxel v = iftGetAdjacentVoxel(A, u, j);

                if(iftValidVoxel(border_gt, v)) {
                    Z->sample[i].feat[j] = iftImgVoxelVal(descendants, v);
                }
            }

            if(!inner_node) {
                if(label->val[p] != gt->val[p]) {
                    Z->sample[i].truelabel = IFT_LEAKING_CLASS;
                } else {
                    if(iftMaxFloatArray(Z->sample[i].feat, Z->nfeats) > 0)
                        iftPrintFloatArray(Z->sample[i].feat, Z->nfeats);

                    Z->sample[i].truelabel = IFT_BORDER_CLASS;
                }
            } else if(use_inner_nodes) {
                if(label->val[p] != gt->val[p]) {
                    Z->sample[i].truelabel = IFT_INTERIOR_ERROR_CLASS;
                } else {
                    Z->sample[i].truelabel = IFT_INTERIOR_CORRECT_CLASS;
                }
            }
            truelabels[Z->sample[i].truelabel-1]++;

            nclasses = iftMax(nclasses, Z->sample[i].truelabel);

            i++;
        }
    }

    Z->nclasses = nclasses;

    iftPrintIntArray(truelabels, 4);

    iftDestroyImage(&border_gt);
    iftDestroyAdjRel(&B);

    return Z;
}
