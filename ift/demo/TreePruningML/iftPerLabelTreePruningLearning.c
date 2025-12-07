#include "ift.h"

iftDict *iftGetArguments(int argc, const char *argv[]);
iftLabeledSet* iftWaterGrayForestFromGT(iftImageForest *fst, iftImage *marker, iftImage *gt);
iftFloatArray * iftLabelRadiusFromSeeds(iftImage *label, iftLabeledSet *seeds);
iftSet* iftWatershedOnMask(iftImageForest *fst, iftLabeledSet *Seeds, iftBMap *mask);
void iftResetProcessedVoxels(iftImageForest *fst, iftSet **processed);
iftImage *iftFilterSAMImage(iftImage *img, double gaussian_radius, double sigma, int radius0, int radius1);
void iftSelectLeakingPointsFromMaximalLeakingPath(iftImage *descendants, iftImage *pred, iftImage *gt,
                                                  iftAdjRel *A, int root_vx, double descendant_drop_perc,
                                                  iftSet **gt_border_leaking, iftSet **burst_leaking);

iftIntArray* iftMapRootLabelsToGT(iftImage *pred, iftImage *label, iftImage *gt);
iftIntArray* iftMapSeedLabelsToGT(iftImage *pred, iftLabeledSet *Seeds, iftImage *label, iftImage *gt);
iftImage* iftRemapLabels(iftImage *label, iftIntArray *label_mapping);
void iftRemapLabeledSet(iftLabeledSet *Seeds, iftIntArray *label_mapping);
iftLabeledSet *iftRootVoxelsToLabeledSet(iftImage *pred, iftImage *label);
iftLabeledSet *iftSortLabeledSetByLabel(iftLabeledSet *Seeds);

int main(int argc, const char *argv[])
{
    iftImageForest        *fst = NULL;
    iftImage        *img=NULL,*basins = NULL, *gt = NULL, *gt_border = NULL;
    iftImage        *marker=NULL;
    iftAdjRel       *A=NULL, *B=NULL, *Amask = NULL;
    timer           *t1=NULL,*t2=NULL;
    iftDict         *args = NULL;
    bool bkg_forbidden;
    iftLabeledSet *Seeds = NULL;
    iftBMap         *mask = NULL, *set_of_interest = NULL;
    iftLabeledSet *S = NULL;
    double min_user_sel_radius = 0.0, max_user_sel_radius = IFT_INFINITY_DBL;
    bool labels_with_at_least_two_seeds;

    /*--------------------------------------------------------*/
    size_t MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/


    args = iftGetArguments(argc, argv);

    img = iftReadImageByExt(iftGetConstStrValFromDict("--input-img", args));
    gt = iftReadImageByExt(iftGetConstStrValFromDict("--gt", args));

    bkg_forbidden = iftDictContainKey("--bkg-forbidden", args, NULL);
    labels_with_at_least_two_seeds = iftDictContainKey("--labels-with-at-least-two-seeds", args, NULL);
    iftPrintDict(args);


    /* the operation is connected for the topology defined by A: A must
       be the same in all operators (including
       iftVolumeClose?). Otherwise, this is not a connected operation in
       the desired topology. */

    if(iftIs3DImage(img)) {
        A = iftSpheric(1.0);
        B = iftSpheric(1.0);
    } else {
        A = iftCircular(1.5);
        B = iftCircular(1.0);
    }

    double descendant_drop_perc = iftGetDblValFromDict("--descendant-drop-perc", args);
    double smooth_radius = (iftDictContainKey("--smooth-radius", args, NULL)) ? iftGetDblValFromDict("--smooth-radius", args) : 3.0;
    double sigma = (iftDictContainKey("--smooth-sigma", args, NULL)) ? iftGetDblValFromDict("--smooth-sigma", args) : 5.0;
    int vol_close_radius0 = (int)((iftDictContainKey("--vol-close-radiusi", args, NULL)) ? iftGetLongValFromDict("--vol-close-radiusi", args) : 3);
    int vol_close_radius1 = (int)((iftDictContainKey("--vol-close-radiusf", args, NULL)) ? iftGetLongValFromDict("--vol-close-radiusf", args) : 4);

    iftImage *tmp = iftFilterSAMImage(img, smooth_radius, sigma, vol_close_radius0, vol_close_radius1);
    iftDestroyImage(&img);
    img = tmp;

    if(iftDictContainKey("--basins", args, NULL)) {
        basins = iftReadImageByExt(iftGetConstStrValFromDict("--basins", args));

        iftImage *tmp = iftFilterSAMImage(basins, smooth_radius, sigma, vol_close_radius0, vol_close_radius1);
        iftDestroyImage(&basins);
        basins = tmp;

        iftWriteImageByExt(img, "filtered.scn");
    } else {
        basins = iftImageBasins(img, A);
    }

    gt_border = iftObjectBorders(gt, B, false, true);

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

    if(iftIs3DImage(img))
        iftWriteImageByExt(marker,"marker.scn");
    else
        iftWriteImageByExt(marker, "marker.pgm");

    if(iftDictContainKey("--min-radius", args, NULL))
        min_user_sel_radius = iftGetDblValFromDict("--min-radius", args);

    if(iftDictContainKey("--max-radius", args, NULL))
        max_user_sel_radius = iftGetDblValFromDict("--max-radius", args);


    iftWaterGrayForest(fst, marker);

    iftImage *watergray_label = iftCopyImage(fst->label);

//    Seeds = iftGeodesicCenters(fst->label);
    Seeds = iftRootVoxelsToLabeledSet(fst->pred, fst->label);

    iftIntArray *label_mapping = iftMapSeedLabelsToGT(fst->pred, Seeds, fst->label, gt);

    iftLabeledSet *sorted_seeds = iftSortLabeledSetByLabel(Seeds);
    iftDestroyLabeledSet(&Seeds);
    Seeds = sorted_seeds;

    iftImage *mapped_labels = iftRemapLabels(fst->label, label_mapping);
    iftDestroyImage(&fst->label);
    fst->label = mapped_labels;


    iftDestroyIntArray(&label_mapping);

    if(iftIs3DImage(img))
        iftWriteImageByExt(fst->label,"mapped_label.scn");
    else
        iftWriteImageByExt(fst->label, "mapped_label.pgm");

    fprintf(stderr, "Number of seeds %d\n", iftLabeledSetSize(Seeds));


    t2     = iftToc();

    fprintf(stdout,"WaterGray in %f ms with %d regions\n",iftCompTime(t1,t2),iftMaximumValue(fst->label));


    iftIntArray *nseeds_per_label = iftCreateIntArray(iftMaximumValue(gt)+1);

    // Creating the seed set from the predecessor map
    for(S = Seeds; S != NULL; S = S->next)
    {
        if(gt->val[S->elem] != S->label) {
            fprintf(stderr,"Seed label %5d (%4d %4d %4d) differs from GT %5d\n",
                    S->label,
                    iftGetVoxelCoord(img, S->elem).x, iftGetVoxelCoord(img, S->elem).y,
                    iftGetVoxelCoord(img, S->elem).z,
                    gt->val[S->elem]);
        }

        nseeds_per_label->val[S->label]++;
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

    fprintf(stdout, "Number of labels with at least two seeds: %03d/%03d\nNumber of labels without seeds: %03d/%03d\nAverage number of seeds per label %f. Label with maximum number of seeds %d: %d\n",
            nlabels_with_two_seeds, nseeds_per_label->n, nlabels_without_seeds, nseeds_per_label->n,
            iftSumIntArray(nseeds_per_label->val, nseeds_per_label->n) / (float)nseeds_per_label->n,
            iftArgMax(nseeds_per_label->val, nseeds_per_label->n),
            nseeds_per_label->val[iftArgMax(nseeds_per_label->val, nseeds_per_label->n)]);


    iftFloatArray *radii = iftLabelRadiusFromSeeds(gt, Seeds);
    float median_radius = iftMedianFloatArray(radii->val, radii->n);
    float max_radius = iftMaxFloatArray(radii->val, radii->n);
    float mean_radius = iftMeanFloatArray(radii->val, radii->n);
    float selected_radius = median_radius*2.0;

    fprintf(stderr, "Max radius %f, mean %f, median %f\nSelected %f\n\n", max_radius, mean_radius, median_radius,  selected_radius);
    if(iftIs3DImage(img)) {
        Amask = iftSphericUnsorted(selected_radius + 1.0);
    } else {
        Amask = iftCircular(selected_radius + 1.0);
    }

    mask = iftCreateBMap(img->n);
    set_of_interest = iftCreateBMap(img->n);

    iftResetImageForest(fst);
    int j = 0;

    for(S = Seeds; S != NULL; S = S->next) {

        if(radii->val[S->label] >= min_user_sel_radius && radii->val[S->label] <= max_user_sel_radius
                && (!labels_with_at_least_two_seeds || nseeds_per_label->val[S->label] > 1)) {
            iftLabeledSet *Scurrent = NULL;
            iftInsertLabeledSet(&Scurrent, S->elem, S->label);

            iftVoxel u = iftGetVoxelCoord(img, S->elem);

            fprintf(stderr, "Seed (%4d %4d %4d) Index %03d Label %5d radius %f. Number of seeds per label %2d\n", u.x, u.y, u.z, j, S->label, radii->val[S->label],
                    nseeds_per_label->val[S->label]);

            iftFillBMap(mask, 0);
            iftFillBMap(set_of_interest, 0);

            for (int i = 0; i < Amask->n; i++) {
                iftVoxel v = iftGetAdjacentVoxel(Amask, u, i);

                if (iftValidVoxel(img, v)) {
                    int q = iftGetVoxelIndex(img, v);
                    iftBMapSet1(mask, q);

                    if (iftVoxelDistance(u, v) >= selected_radius) {
                        iftBMapSet1(set_of_interest, q);
                    }
                }
            }

            iftSet *processed = iftWatershedOnMask(fst, Scurrent, mask);

            iftImage *descendants = iftDescendantMapOnMask(fst->pred, fst->A, mask, Scurrent, set_of_interest);


            for (int p = 0; p < gt->n; p++) {
                if (gt->val[p] == S->label) {
                    fst->label->val[p] = 2 * S->label;
                }
            }

            for (int p = 0; p < gt->n; p++) {
                if (iftBMapValue(set_of_interest, p)) {
                    fst->label->val[p] = 3 * S->label;
                }
            }

            iftSet *burst_voxel = NULL, *gt_leaking = NULL;

            iftSelectLeakingPointsFromMaximalLeakingPath(descendants, fst->pred, gt, A, S->elem, descendant_drop_perc,
                                                         &gt_leaking, &burst_voxel);

            for (int i = 0; i < A->n; i++) {
                iftVoxel v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(img, v)) {
                    fst->label->val[iftGetVoxelIndex(img, v)] = 4 * S->label;
                }
            }
            if(burst_voxel != NULL) {
                u = iftGetVoxelCoord(img, burst_voxel->elem);

                fprintf(stderr, "Burst leaking voxel at (%4d %4d %4d). Descendant count %5d predecessor descendant count %5d\n", u.x, u.y, u.z,
                        descendants->val[burst_voxel->elem], (fst->pred->val[burst_voxel->elem] != IFT_NIL) ? descendants->val[fst->pred->val[burst_voxel->elem]] : IFT_NIL);

                for (int i = 0; i < A->n; i++) {
                    iftVoxel v = iftGetAdjacentVoxel(A, u, i);
                    if (iftValidVoxel(img, v)) {
                        fst->label->val[iftGetVoxelIndex(img, v)] = 5 * S->label;
                    }
                }
            } else {
                fprintf(stderr, "No burst leaking voxel");
            }

            if(gt_leaking != NULL) {
                u = iftGetVoxelCoord(img, gt_leaking->elem);

                fprintf(stderr, "GT leaking voxel at (%4d %4d %4d). Descendant count %5d predecessor descendant count %5d\n", u.x, u.y, u.z,
                        descendants->val[gt_leaking->elem], (fst->pred->val[gt_leaking->elem] != IFT_NIL) ? descendants->val[fst->pred->val[gt_leaking->elem]] : IFT_NIL);

                for (int i = 0; i < A->n; i++) {
                    iftVoxel v = iftGetAdjacentVoxel(A, u, i);
                    if (iftValidVoxel(img, v)) {
                        fst->label->val[iftGetVoxelIndex(img, v)] = 6 * S->label;
                    }
                }
            } else {
                fprintf(stderr, "No GT leaking voxel\n");
            }
            fprintf(stderr,"\n\n");


            iftWriteImageByExt(fst->label, "single_label_%05d_%05d.scn", S->label, j);
            iftWriteImageByExt(descendants, "single_label_%05d_%05d_desc.scn", S->label, j);

            iftResetProcessedVoxels(fst, &processed);
            iftDestroyLabeledSet(&Scurrent);
            iftDestroyImage(&descendants);
            iftDestroySet(&burst_voxel);
            iftDestroySet(&gt_leaking);

            j++;
        }
    }

    iftDestroyBMap(&mask);
    iftDestroyBMap(&set_of_interest);
    iftDestroyAdjRel(&Amask);
    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
    iftDestroyImage(&img);
    iftDestroyImage(&gt);
    iftDestroyImage(&gt_border);
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyImageForest(&fst);
    iftDestroyLabeledSet(&Seeds);
    iftDestroyDict(&args);
    iftDestroyImage(&watergray_label);

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
            {.short_name = "", .long_name = "--descendant-drop-perc", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=false, .help="Percentage of drop in descendant count to consider a leaking voxel on a burst zone"},
            {.short_name = "", .long_name = "--height", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="Height used to implicitly compute the H-minima transform"},
            {.short_name = "", .long_name = "--volume", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="Value used to close basins by volume"},
            {.short_name = "", .long_name = "--bkg-forbidden", .has_arg=false, .arg_type=IFT_BOOL_TYPE,
                    .required=false, .help="Forbids competition to occur in the background"},
            {.short_name = "", .long_name = "--labels-with-at-least-two-seeds", .has_arg=false, .arg_type=IFT_BOOL_TYPE,
                    .required=false, .help="Only selects labels with two or more seeds"},
            {.short_name = "", .long_name = "--min-radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=false, .help="Minimum radius to be considered"},
            {.short_name = "", .long_name = "--max-radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                .required=false, .help="Maximum radius to be considered"},
            {.short_name = "", .long_name = "--smooth-sigma", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=false, .help="Gaussian smoothing sigma value"},
            {.short_name = "", .long_name = "--vol-close-radiusi", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="First radius of iterative closing"},
            {.short_name = "", .long_name = "--vol-close-radiusf", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="Last radius of iterative closing"},
    };
    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


iftImage *iftFilterSAMImage(iftImage *img, double gaussian_radius, double sigma, int radius0, int radius1) {
    iftAdjRel *A = iftSpheric(gaussian_radius);
    iftImage *smoothed = iftSmoothImage(img, A, sigma);
    iftImage *closed = iftVolumeClose(smoothed, radius0);

    for(int r = radius0+1; r <= radius1; r++) {
        iftImage *tmp = iftVolumeClose(closed, r);
        iftDestroyImage(&closed);
        closed = tmp;
    }

    iftDestroyImage(&smoothed);
    iftDestroyAdjRel(&A);

    return closed;
}

//iftLabeledSet* iftWaterGrayForestFromGT(iftImageForest *fst, iftImage *marker, iftImage *gt)
//{
//    iftImage   *label=fst->label, *basins=fst->img, *pathval=fst->pathval;
//    iftImage   *pred=fst->pred, *root=fst->root;
//    iftGQueue  *Q=fst->Q;
//    iftAdjRel  *A=fst->A;
//    int         i,p,q,tmp;
//    iftVoxel    u,v;
//    iftLabeledSet *seeds = NULL;
//
//    // Initialization
//
//    for (p=0; p < pathval->n; p++) {
//        if(gt->val[p] > 0) {
//            pathval->val[p] = marker->val[p] + 1;
//            iftInsertGQueue(&Q, p);
//        }
//    }
//    // Image Foresting Transform
//    while(!iftEmptyGQueue(Q)) {
//        p=iftRemoveGQueue(Q);
//
//        if (pred->val[p] == IFT_NIL) { // root voxel
//            pathval->val[p]  -= 1;
//            label->val[p]=gt->val[p];
//            iftInsertLabeledSet(&seeds, p, label->val[p]);
//        }
//
//        u = iftGetVoxelCoord(basins,p);
//
//        for (i=1; i < A->n; i++){
//            v = iftGetAdjacentVoxel(A,u,i);
//            if (iftValidVoxel(basins,v)){
//                q = iftGetVoxelIndex(basins,v);
//                if (gt->val[q] > 0 && pathval->val[q] > pathval->val[p]){
//                    tmp = iftMax(pathval->val[p], basins->val[q]);
//                    if (tmp < pathval->val[q]){
//                        iftRemoveGQueueElem(Q,q);
//                        label->val[q]      = label->val[p];
//                        root->val[q]       = root->val[p];
//                        pred->val[q]       = p;
//                        pathval->val[q]    = tmp;
//                        iftInsertGQueue(&Q, q);
//                    }
//                }
//            }
//        }
//    }
//
//    return seeds;
//}


/**
 * @brief Computes the radius necessary to draw a sphere around a label centered at one of its seeds, and returns
 * the corresponding value for each label in a float array.
 *
 * @param label Input segmentation label (i.e., the ground truth)
 * @param seeds Input seeds
 * @return
 */
iftFloatArray* iftLabelRadiusFromSeeds(iftImage *label, iftLabeledSet *seeds) {
    iftIntMatrix *seed_locations = NULL;
    int max_lb = iftMaximumValue(label);
    int nseeds = iftLabeledSetSize(seeds);
    iftFloatArray *radii = NULL;

    seed_locations  = iftCreateIntMatrix(nseeds, max_lb + 1);
    radii           = iftCreateFloatArray(max_lb + 1);

    for(int i = 0; i < seed_locations->n; i++) {
        seed_locations->val[i] = IFT_NIL;
    }

    for(iftLabeledSet *S = seeds; S != NULL; S = S->next) {
        for(int i = 0; i < seed_locations->ncols; i++) {
            if(iftMatrixElem(seed_locations, i, S->label) < 0) {
                iftMatrixElem(seed_locations, i, S->label) = S->elem;
                break;
            }
        }
    }

    for(int p = 0; p < label->n; p++) {
        if(label->val[p] > 0) {
            iftVoxel u = iftGetVoxelCoord(label, p);
            for(int i = 0; i < seed_locations->ncols; i++) {
                if(iftMatrixElem(seed_locations, i, label->val[p]) >= 0) {
                    iftVoxel v = iftGetVoxelCoord(label, iftMatrixElem(seed_locations, i, label->val[p]));
                    float dist = iftVoxelDistance(u, v);

                    radii->val[label->val[p]] = iftMax(dist, radii->val[label->val[p]]);
                } else {
                    break;
                }
            }
        }
    }

    iftDestroyIntMatrix(&seed_locations);

    return radii;
}


iftSet* iftWatershedOnMask(iftImageForest *fst, iftLabeledSet *Seeds, iftBMap *mask)
{
    iftImage   *label=fst->label, *basins=fst->img, *pathval=fst->pathval;
    iftImage   *pred=fst->pred, *root=fst->root, *marker=fst->marker;
    iftGQueue  *Q=fst->Q;
    iftAdjRel  *A=fst->A;
    int         i,p,q,tmp;
    iftVoxel    u,v;
    iftSet *processed = NULL;

    // Initialization
    iftLabeledSet *S = Seeds;
    while (S != NULL)
    {
        p = S->elem;
        if(iftBMapValue(mask, p)) {
            label->val[p]   = S->label;
            marker->val[p]  = S->marker;
            pathval->val[p] = 0;
            root->val[p]    = p;
            pred->val[p]    = IFT_NIL;
            iftInsertGQueue(&Q, p);
            S = S->next;
        }
    }

    // Image Foresting Transform

    while(!iftEmptyGQueue(Q)) {
        p=iftRemoveGQueue(Q);

        u = iftGetVoxelCoord(basins,p);

        iftInsertSet(&processed, p);

        for (i=1; i < A->n; i++){
            v = iftGetAdjacentVoxel(A,u,i);
            if (iftValidVoxel(basins,v)){
                q = iftGetVoxelIndex(basins,v);
                if (iftBMapValue(mask, q) && pathval->val[q] > pathval->val[p]) {
                    tmp = iftMax(pathval->val[p], basins->val[q]);
                    if (tmp < pathval->val[q]){
                        iftRemoveGQueueElem(Q,q);
                        label->val[q]       = label->val[p];
                        root->val[q]        = root->val[p];
                        pred->val[q]        = p;
                        marker->val[q]      = marker->val[p];
                        pathval->val[q]    = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    return processed;
}

void iftResetProcessedVoxels(iftImageForest *fst, iftSet **processed) {
    fst->Q->C.minvalue = IFT_INFINITY_INT;
    fst->Q->C.maxvalue = IFT_INFINITY_INT_NEG;

    for (int i=0; i < fst->Q->C.nbuckets+1; i++)
        fst->Q->C.first[i] = fst->Q->C.last[i] = IFT_NIL;

    // Resetting only what was processed
    while(processed != NULL && *processed != NULL) {
        int i = iftRemoveSet(processed);
        fst->Q->L.elem[i].next  = fst->Q->L.elem[i].prev = IFT_NIL;
        fst->Q->L.elem[i].color = IFT_WHITE;
        fst->pathval->val[i]     = IFT_INFINITY_INT;
        fst->pred->val[i]        = IFT_NIL;
        fst->label->val[i]       = 0;
        fst->root->val[i]        = i;
        fst->marker->val[i]      = IFT_NIL;
    }
}

void iftSelectLeakingPointsFromMaximalLeakingPath(iftImage *descendants, iftImage *pred, iftImage *gt,
                                                  iftAdjRel *A, int root_vx, double descendant_drop_perc,
                                                  iftSet **gt_border_leaking, iftSet **burst_leaking) {
    int next_voxel = root_vx;

    if(pred->val[root_vx] != IFT_NIL) {
        iftError("Specified voxel (%d %d %d) is not a root ", "iftSelectLeakingPointsFromMaximalLeakingPath",
                 iftGetVoxelCoord(pred, root_vx).x, iftGetVoxelCoord(pred, root_vx).y,
                 iftGetVoxelCoord(pred, root_vx).z);
    }

    *gt_border_leaking = NULL;
    *burst_leaking = NULL;

    while(next_voxel >= 0) {
        int p = next_voxel;

        if(pred->val[p] != IFT_NIL) {
            // Selecting the first voxel along the leaking path that leaves the object as a leaking point.
            if(*gt_border_leaking == NULL) {
                if(gt->val[p] != gt->val[pred->val[p]]) {
                    fprintf(stderr, "GT leaking detected %8d at (%8d %8d %8d) GT Label %5d -- pred count %8d at (%8d %8d %8d) GT Label %5d\n", descendants->val[p],
                            iftGetVoxelCoord(pred, p).x, iftGetVoxelCoord(pred, p).y, iftGetVoxelCoord(pred, p).z, gt->val[p],
                            descendants->val[pred->val[p]], iftGetVoxelCoord(pred, pred->val[p]).x, iftGetVoxelCoord(pred, pred->val[p]).y,
                            iftGetVoxelCoord(pred, pred->val[p]).z, gt->val[pred->val[p]]);

                    iftInsertSet(gt_border_leaking, pred->val[p]);
                }
            }

            // If the number of descendants of p relative to the number of descendants of its predecessor
            // falls by more than the specified percentage, we consider pred->val[p] a leaking burst voxel.
            // We are currently only selecting the *first* burst
            if(*burst_leaking == NULL) {
                if(1.0 - (descendants->val[p] / (double) descendants->val[pred->val[p]]) >= descendant_drop_perc) {
                    fprintf(stderr, "Burst detected %d at (%d %d %d) -- pred count %d at (%d %d %d)\n", descendants->val[p],
                    iftGetVoxelCoord(pred, p).x, iftGetVoxelCoord(pred, p).y, iftGetVoxelCoord(pred, p).z,
                    descendants->val[pred->val[p]], iftGetVoxelCoord(pred, pred->val[p]).x, iftGetVoxelCoord(pred, pred->val[p]).y,
                            iftGetVoxelCoord(pred, pred->val[p]).z);

                    iftInsertSet(burst_leaking, pred->val[p]);
                }
            }
        }

        fprintf(stderr, "Voxel (%4d %4d %4d) Predecessor %8d Descendant count %5d Predecessor descendant count %5d GT Label %3d Predecessor GT Label %3d\n",
                iftGetVoxelCoord(pred, p).x, iftGetVoxelCoord(pred, p).y, iftGetVoxelCoord(pred, p).z,
                pred->val[p], descendants->val[p], (pred->val[p] == IFT_NIL) ? IFT_NIL : descendants->val[pred->val[p]], gt->val[p],
                (pred->val[p] == IFT_NIL) ? IFT_NIL : gt->val[pred->val[p]]);

        iftVoxel u = iftGetVoxelCoord(pred, p);
        int max_num_descendants = IFT_INFINITY_INT_NEG;
        next_voxel = IFT_NIL;

        // Finding the next voxel in the optimum path with highest descendant count (i.e., walking along the leaking path)
        for(int i = 1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if(iftValidVoxel(pred, v)) {
                int q = iftGetVoxelIndex(pred, v);

                if(pred->val[q] == p) {
                    if (descendants->val[q] > max_num_descendants) {
                        max_num_descendants = descendants->val[q];
                        next_voxel = q;
                    }
                }
            }
        }
    }
}

iftIntArray* iftMapRootLabelsToGT(iftImage *pred, iftImage *label, iftImage *gt) {
    iftIntArray *label_mapping = NULL;

    label_mapping = iftCreateIntArray(iftMaximumValue(label)+1);

    for(int p = 0; p < pred->n; p++) {
        if(pred->val[p] == IFT_NIL) {
            label_mapping->val[label->val[p]] = gt->val[p];
        }
    }

    return label_mapping;
}


iftIntArray* iftMapSeedLabelsToGT(iftImage *pred, iftLabeledSet *Seeds, iftImage *label, iftImage *gt) {
    iftIntArray *label_mapping = NULL;

    label_mapping = iftCreateIntArray(iftMaximumValue(label)+1);

    for(iftLabeledSet *S = Seeds; S != NULL; S = S->next) {
        label_mapping->val[S->label] = gt->val[S->elem];
        S->label = label_mapping->val[S->label];
    }

    return label_mapping;
}

iftImage* iftRemapLabels(iftImage *label, iftIntArray *label_mapping) {
    iftImage *mapped_label = iftCopyImage(label);

    for(int p = 0; p < label->n; p++) {
        mapped_label->val[p] = label_mapping->val[label->val[p]];
    }

    return mapped_label;
}

void iftRemapLabeledSet(iftLabeledSet *Seeds, iftIntArray *label_mapping) {
    for(iftLabeledSet *S = Seeds; S != NULL; S = S->next) {
        S->label = label_mapping->val[S->label];
    }
}

iftLabeledSet *iftRootVoxelsToLabeledSet(iftImage *pred, iftImage *label)
{
    iftLabeledSet *S = NULL;

    for (int p=0; p < pred->n; p++) {
        if (pred->val[p] == IFT_NIL){
            iftInsertLabeledSet(&S, p, label->val[p]);
        }
    }

    return S;
}

iftLabeledSet *iftSortLabeledSetByLabel(iftLabeledSet *Seeds) {
    int nseeds = iftLabeledSetSize(Seeds);
    iftIntArray *labels = NULL;
    iftIntArray *elems = NULL;
    iftLabeledSet *sorted_seeds = NULL;

    labels = iftCreateIntArray(nseeds);
    elems  = iftCreateIntArray(nseeds);

    size_t i = 0;
    for(iftLabeledSet *S = Seeds; S != NULL; S = S->next) {
        labels->val[i] = S->label;
        elems->val[i]  = S->elem;

        i++;
    }

    iftQuickSort(labels->val, elems->val, 0, labels->n - 1, IFT_DECREASING);

    for(i = 0; i < labels->n; i++) {
        iftInsertLabeledSet(&sorted_seeds, elems->val[i], labels->val[i]);
    }
    iftDestroyIntArray(&labels);
    iftDestroyIntArray(&elems);

    return sorted_seeds;
}