#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(iftDict *args, char **img_path, char **markers_path, char **out_img_path, int *k, float *adj_rel_r);
iftImage *iftDynamicIFTPathValue(iftMImage *mimg, iftLabeledSet *seeds, iftAdjRel *A, int max_pred_length);
iftImage *iftWatershedPathValue(  iftImage *basins, iftAdjRel *Ain, iftLabeledSet *seeds);
iftImage* iftDetectRealTieZones(iftImage* tiesMap, iftImage* objCost, iftImage* bkgCost);
iftImage *iftMaskOfEqualValues(iftImage* value1, iftImage* value2);
/*************************************************************/


int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // mandatory args
    char *img_path      = NULL;
    char *markers_path  = NULL;
    char *out_img_path = NULL;
    char *out_ext = NULL, *out_base_path = NULL;
    int k;
    float adj_rel_r;

    iftGetRequiredArgs(args, &img_path, &markers_path, &out_img_path, &k, &adj_rel_r);

    timer *t1 = iftTic();

    puts("- Reading Input Image");
    iftImage *img = iftReadImageByExt(img_path);
    iftImage *basins = NULL, *obj_cost = NULL, *bkg_cost = NULL;
    iftImage *tie_img = NULL, *equal_cost = NULL;
    iftMImage *mimg = NULL;
    iftAdjRel *A = NULL, *B = NULL;

    if(iftIs3DImage(img)){
        A = iftSpheric(adj_rel_r);
        B = iftSpheric(sqrtf(2.0f)); // Basins AdjRel
    } else {
        A = iftCircular(adj_rel_r);
        B = iftCircular(sqrtf(2.0f)); // Basin AdjRel
    }

    if (iftIsColorImage(img)){
        mimg = iftImageToMImage(img, LABNorm_CSPACE);
    } else {
        mimg = iftImageToMImage(img, GRAY_CSPACE);
    }

    out_ext = iftFileExt(out_img_path);
    out_base_path = iftBasename(out_img_path);

    puts("- Reading Markers");
    iftLabeledSet *seeds = iftReadSeeds(img, markers_path);
    iftLabeledSet *obj_seeds = iftCopyLabels(seeds, 1);
    iftLabeledSet *bkg_seeds = iftCopyLabels(seeds, 0);
    iftDestroyLabeledSet(&seeds);

    puts("- Generating binary mask for Watershed tie zones");
    basins = iftMImageBasins(mimg, B);
    iftDestroyAdjRel(&B);

    obj_cost = iftWatershedPathValue(basins, A, obj_seeds);
    bkg_cost = iftWatershedPathValue(basins, A, bkg_seeds);
    equal_cost = iftMaskOfEqualValues(obj_cost, bkg_cost);
    tie_img = iftDetectRealTieZones(equal_cost, obj_cost, bkg_cost);
    char *watershed_out_path = iftConcatStrings(3, out_base_path, "_watershed", out_ext);
    iftWriteImageByExt(tie_img, watershed_out_path);

    iftFree(watershed_out_path);
    iftDestroyImage(&obj_cost);
    iftDestroyImage(&bkg_cost);
    iftDestroyImage(&equal_cost);
    iftDestroyImage(&tie_img);
    iftDestroyImage(&basins);

    puts("- Generating binary mask for DynamicIFT tie zones");

    obj_cost = iftDynamicIFTPathValue(mimg, obj_seeds, A, k);
    bkg_cost = iftDynamicIFTPathValue(mimg, bkg_seeds, A, k);
    equal_cost = iftMaskOfEqualValues(obj_cost, bkg_cost);
    tie_img = iftDetectRealTieZones(equal_cost, obj_cost, bkg_cost);
    char *dynamic_out_path = iftConcatStrings(3, out_base_path, "_dynamic", out_ext);
    iftWriteImageByExt(tie_img, dynamic_out_path);

    iftFree(dynamic_out_path);
    iftDestroyImage(&obj_cost);
    iftDestroyImage(&bkg_cost);
    iftDestroyImage(&equal_cost);
    iftDestroyImage(&tie_img);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftFree(img_path);
    iftFree(out_img_path);
    iftFree(markers_path);
    iftFree(out_base_path);
    iftDestroyLabeledSet(&obj_seeds);
    iftDestroyLabeledSet(&bkg_seeds);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
    iftDestroyAdjRel(&A);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Return binary mask of Watershed Tranform and Dynamic IFT Tie zones\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input 2D Image to be segmented."},
        {.short_name = "-m", .long_name = "--markers", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Markers on input image."},
        {.short_name = "-o", .long_name = "--output-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Path of the Output Segmentation Mask."},
        {.short_name = "-k", .long_name = "--k-max", .has_arg=true, .arg_type=IFT_LONG_TYPE,
         .required=true, .help="Max quantity of nodes on predecessor set."},
        {.short_name = "-a", .long_name = "--adjacency-relation", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                .required=true, .help="Propagation adjacency relation."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(iftDict *args, char **img_path, char **markers_path, char **out_img_path, int *k, float *adj_rel_r) {
    *img_path      = iftGetStrValFromDict("--input-image", args);
    *markers_path  = iftGetStrValFromDict("--markers", args);
    *out_img_path = iftGetStrValFromDict("--output-mask", args);
    *k = (int) iftGetLongValFromDict("--k-max", args);
    *adj_rel_r = (float) iftGetDblValFromDict("--adjacency-relation", args);

    char *parent_dir = iftParentDir(*out_img_path);
    if (!iftDirExists(parent_dir))
        iftMakeDir(parent_dir);
    iftFree(parent_dir);

    puts("-----------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Markers: \"%s\"\n", *markers_path);
    printf("- Adjacency Relation Radius: \"%.3f\"\n", *adj_rel_r);
    printf("- k: \"%d\"\n", *k);
    printf("- Output Image: \"%s\"\n", *out_img_path);
    puts("-----------------------\n");
}
/*************************************************************/


iftImage *iftDynamicIFTPathValue(iftMImage *mimg, iftLabeledSet *seeds, iftAdjRel *A, int max_pred_length){

    iftImage *label    = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    iftImage *pathval     = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    iftImage *pathval_dyn  = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    iftImage *root     = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    iftIntArray **forest = iftAlloc( (size_t) mimg->n, sizeof(iftIntArray*));
    iftIntArray *aux_array = iftCreateIntArray( (size_t) max_pred_length);

    // Falcao
    int mimg_max = (int) 2 * (iftMMaximumValue(mimg, -1) * 100);//sqrt(mimg->m);

    /** Initialization */
    iftGQueue *Q      = iftCreateGQueue(mimg_max + 1, mimg->n, pathval->val);
    iftGQueue *Q_dyn   = iftCreateGQueue(mimg_max + 1, mimg->n, pathval_dyn->val);

    for (int p = 0; p < mimg->n; p++){
        pathval->val[p]    = IFT_INFINITY_INT;
        pathval_dyn->val[p] = IFT_INFINITY_INT;
    }

    for(iftLabeledSet *L = seeds; L != NULL; L = L->next){
        int p = L->elem;

        pathval->val[p]    = 0;
        pathval_dyn->val[p] = 0;
        label->val[p]   = L->label;
        root->val[p] = p;

        forest[p] = iftCreateIntArray((size_t) max_pred_length);
        forest[p]->n = 0;

        iftInsertGQueue(&Q, p);
    }
    /*******/

    // ift
    while (!iftEmptyGQueue(Q)) {
        int p      = iftRemoveGQueue(Q);
        iftVoxel u = iftMGetVoxelCoord(mimg, p);

        int r = root->val[p];

        if(forest[r]->n < max_pred_length) {
            forest[r]->val[forest[r]->n] = p;
            forest[r]->n++;
            _iftSortedUpdateSubtreeArray(mimg, forest[r], aux_array, pathval_dyn, Q_dyn);

            /* Given a forest root x, I(x) is the mean of the whole tree */
//            int x = forest[r]->val[0];
//            int y = forest[r]->val[ forest[r]->n - 1];
//            for(int i = 0; i < mimg->m; i++){
//                mimg->band[i].val[x] = iftRound((mimg->band[i].val[x] * (forest[r]->n - 1) +
//                                       mimg->band[i].val[y]) / forest[r]->n);
//            }
        }

        for (int i = 1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(mimg, v)) {
                int q = iftMGetVoxelIndex(mimg, v);

                if (Q->L.elem[q].color != IFT_BLACK && pathval->val[p] < pathval->val[q]) {
                    // p conquers q
                    int arc_w = _iftArcWeightByIntArray(mimg, q, forest[root->val[p]], pathval_dyn);

                    int tmp = iftMax(pathval->val[p], arc_w);

                    if (tmp < pathval->val[q]) {

                        if(Q->L.elem[q].color == IFT_GRAY) {
                            iftRemoveGQueueElem(Q, q);
                        }

                        pathval->val[q]  = tmp;
                        pathval_dyn->val[q] = arc_w;
                        label->val[q] = label->val[p];
                        root->val[q]  = root->val[p];

                        iftInsertGQueue(&Q, q);

                    }
                }
            }
        }
    }

    iftDestroyImage(&label);
    iftDestroyImage(&pathval_dyn);
    iftDestroyGQueue(&Q);
    iftDestroyGQueue(&Q_dyn);
    iftDestroyImage(&root);
    iftDestroyIntArray(&aux_array);

    for(iftLabeledSet *L = seeds; L != NULL; L = L->next){
        int p = L->elem;
        iftDestroyIntArray(&forest[p]);
    }
    iftFree(forest);

    return pathval;
}

iftImage* iftDetectRealTieZones(iftImage* tiesMap, iftImage* objCost, iftImage* bkgCost){

    iftImage* new = iftCopyImage(tiesMap);
    iftFIFO* F = iftCreateFIFO(tiesMap->n);
    iftLIFO* Stack = iftCreateLIFO(tiesMap->n);
    iftAdjRel* A = iftCircular(1.0);

    for(int p = 0; p < tiesMap->n; p++){

        if(F->color[p] == IFT_WHITE && tiesMap->val[p] == 255){

            iftInsertFIFO(F, p);

            int fakeTie = 1;
            int dif_obj_bkg = 0;

            while(!iftEmptyFIFO(F)){

                int q = iftRemoveFIFO(F);
                iftVoxel u = iftGetVoxelCoord(new, q);

                iftInsertLIFO(Stack, q);

                for(int i = 0; i < A->n; i++){
                    iftVoxel v = iftGetAdjacentVoxel(A, u, i);

                    if(iftValidVoxel(new, v)){
                        int w = iftGetVoxelIndex(new, v);

                        if(F->color[w] == IFT_WHITE && tiesMap->val[w] == 255){
                            iftInsertFIFO(F, w);
                        } else if(fakeTie == 1 && tiesMap->val[w] != 255){
                            if(dif_obj_bkg == 0){ /* Essa condição == 0, ocorre apenas uma vez por
                                                   * por componente conexo */
                                dif_obj_bkg = objCost->val[w] - bkgCost->val[w];
                            } else if(dif_obj_bkg < 0 && (objCost->val[w] - bkgCost->val[w]) > 0 ){
                                fakeTie = 0;
                            } else if(dif_obj_bkg > 0 && (objCost->val[w] - bkgCost->val[w]) < 0 ){
                                fakeTie = 0;
                            }
                        }
                    }
                }
            }

            /* Remove fake zones */
            if(fakeTie == 1){
                while(!iftEmptyLIFO(Stack)){
                    int q = iftRemoveLIFO(Stack);
                    new->val[q] = 0;
                }
            }
            iftResetLIFO(Stack);
        }
    }

    iftDestroyFIFO(&F);
    iftDestroyLIFO(&Stack);
    iftDestroyAdjRel(&A);

    return new;
}


iftImage *iftWatershedPathValue(  iftImage *basins, iftAdjRel *Ain, iftLabeledSet *seeds) {
    iftImage  *pathval = NULL, *label = NULL;
    iftGQueue  *Q = NULL;
    int      i, p, q, tmp;
    iftVoxel    u, v;
    iftLabeledSet *S = seeds;

    iftAdjRel *A = NULL;
    if (Ain == NULL) {
        if (iftIs3DImage(basins))
            A = iftSpheric(1.0);
        else A = iftCircular(1.0);
    }
    else A = Ain;

    // Initialization
    pathval  = iftCreateImage(basins->xsize, basins->ysize, basins->zsize);
    label = iftCreateImage(basins->xsize, basins->ysize, basins->zsize);
    Q     = iftCreateGQueue(iftMaximumValue(basins) + 1, basins->n, pathval->val);
    for (p = 0; p < basins->n; p++)
    {
        pathval->val[p] = IFT_INFINITY_INT;
    }

    while (S != NULL)
    {
        p = S->elem;
        label->val[p] = S->label;
        pathval->val[p] = 0;
        iftInsertGQueue(&Q, p);
        S = S->next;
    }


    // Image Foresting Transform

    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(basins, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(basins, v))
            {
                q = iftGetVoxelIndex(basins, v);
                if (pathval->val[q] > pathval->val[p])
                {
                    tmp = iftMax(pathval->val[p], basins->val[q]);
                    if (tmp < pathval->val[q])  // For this path-value function,
                    {
                        // this implies that q has never
                        // been inserted in Q.
                        label->val[q] = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    if (Ain == NULL) {
        iftDestroyAdjRel(&A);
    }
    iftDestroyGQueue(&Q);
    iftDestroyImage(&label);

    return (pathval);
}

iftImage *iftMaskOfEqualValues(iftImage* value1, iftImage* value2){

    iftImage* output = iftCreateImageFromImage(value1);

    for(int p = 0; p < output->n; p++){
        output->val[p] = (value1->val[p] == value2->val[p]) * 255;
    }

    return output;
}