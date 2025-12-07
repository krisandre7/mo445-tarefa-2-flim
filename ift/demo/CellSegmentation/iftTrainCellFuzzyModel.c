//
// Created by tvspina on 11/15/16.
//

#include "ift.h"

typedef struct iftCellFOM {
    iftAdjRel **ALabels;
    iftIntArray *labels;
    iftAdjRel *AUncertain;
    iftFImage *model;
}iftCellFOM;

iftDict *iftGetArguments(int argc, const char *argv[]);

iftImageArray *iftExtractCells(iftImage *label);

iftImageArray *iftFilterObjectsByVolume(iftImageArray *images, int vol_min, int vol_max);

iftImageArray *iftAlignImagesByCentroid(iftImageArray *images, int padding);

iftFImage* iftFuzzyCellModelByAverage(iftImageArray *images);

bool iftHasUncertainNeighbor(  iftFImage *model, float bkg_thresh, float obj_thresh,   iftAdjRel *A, int p);

void iftDrawAdjRel(iftImage *img, iftAdjRel *A, int p, iftColor YCbCr, bool include_center);
void iftSetValueOnAdjRel(iftImage *img, iftAdjRel *A, int p, int value, bool include_center);

/**
 * @brief Creates a Cell FOM from a fuzzy membership image and takes ownership of model.
 *
 * @param model     iftFImage containing the fuzzy model
 * @param A         Adjacency relation considered by the delineation algorithm to determine the seeds.
 * @param bkg_thresh Background threshold used to select background seeds adjacent to the uncertainty region.
 * @param obj_thresh Object threshold used to select object seeds adjacent to the uncertainty region.
 * @return A Cell FOM
 */
iftCellFOM * iftCreateCellFOM(iftFImage *model, iftAdjRel *A, float bkg_thresh, float obj_thresh);

void iftDestroyCellFOM(iftCellFOM **FOM);

/**
 * @brief Allocates iftCellFOM data.
 */
iftCellFOM *iftAllocCellFOM(iftFImage *model, iftAdjRel **ALabels, iftAdjRel *AUncertain, iftIntArray *labels);


iftVoxel iftSegmentSingleCellByWatershedMSPS(iftImage *orig, iftImageForest *fst, iftCellFOM *FOM, iftVoxel cand_pos,
                                             iftBMap *search_region, iftLabeledSet **objects);

iftImage* iftSegmentCellsByWatershed(iftImage *orig, iftCellFOM *FOM, iftAdjRel *A);

void iftSetCellSearchRegion(  iftCellFOM *FOM,   iftImage *label, iftBMap *search_region, iftVoxel cand_position);

iftImage *iftBlackTopHat(iftImage *img, iftAdjRel *A);
iftImage *iftWhiteTopHat(iftImage *img, iftAdjRel *A);

int main(int argc, const char *argv[]) {

    iftFImage *model = NULL;
    iftImage *img = NULL;
    iftImage *label = NULL;
    iftImage *target = NULL;
    iftImageArray *cells = NULL;
    iftImageArray *cells_aligned = NULL;
    iftImageArray *cells_filtered = NULL;

    iftCellFOM *FOM = NULL;
    iftAdjRel *A = NULL;

    int cell_size_min = IFT_INFINITY_INT_NEG, cell_size_max = IFT_INFINITY_INT;
    bool cell_filtering = false;

    iftDict *args = NULL;

    size_t MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    args = iftGetArguments(argc, argv);

    const char *img_fname = iftGetConstStrValFromDict("-i", args);
    const char *label_fname = iftGetConstStrValFromDict("-l", args);
    const char *test_fname = iftGetConstStrValFromDict("-t", args);

    img = iftReadImageByExt(img_fname);
    label = iftReadImageByExt(label_fname);
    target = iftReadImageByExt(test_fname);

    cells = iftExtractCells(label);

    if(iftDictContainKey("--min-vol", args, NULL)) {
        cell_size_min = iftGetLongValFromDict("--min-vol", args);
        cell_filtering = true;
    }


    if(iftDictContainKey("--max-vol", args, NULL)) {
        cell_size_max = iftGetLongValFromDict("--max-vol", args);
        cell_filtering = true;
    }

    if(cell_filtering) {
        cells_filtered = iftFilterObjectsByVolume(cells, cell_size_min, cell_size_max);
    } else {
        cells_filtered = cells;
    }

    cells_aligned = iftAlignImagesByCentroid(cells_filtered, 10);

    model = iftFuzzyCellModelByAverage(cells_aligned);

    if(iftIs3DFImage(model))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.5);

    FOM = iftCreateCellFOM(model, A, 0.0, 0.95);

    iftImage *tmp = iftFImageToImage(model, 255);
    iftWriteImageByExt(tmp, "model%s", iftFileExt(img_fname));

    iftDrawAdjRel(tmp, FOM->ALabels[0], iftGetVoxelIndex(tmp, iftImageCenter(tmp)),
                  iftRGBtoYCbCr(iftRGBColor(255, 255, 0), 255), true);

    iftDrawAdjRel(tmp, FOM->ALabels[1], iftGetVoxelIndex(tmp, iftImageCenter(tmp)),
                  iftRGBtoYCbCr(iftRGBColor(0, 255, 0), 255), true);


    iftWriteImageByExt(tmp, "model_seeds.ppm");

    iftDestroyImage(&tmp);
//
//    tmp = iftFImageToImage(model, 255);
//
//    iftDrawAdjRel(tmp, FOM->AUncertain, iftGetVoxelIndex(tmp, iftImageCenter(tmp)),
//                  iftRGBtoYCbCr(iftRGBColor(0, 0, 255), 255), false);
//
//    iftWriteImageByExt(tmp, "model_uncertain.ppm");
//    iftDestroyImage(&tmp);
//

    iftImage *regmin = iftRegionalMinima(target);
    iftImageForest *fst = iftCreateImageForest(target, A);

    iftVoxel first_seed_vx;

    iftBMap *search_region = iftCreateBMap(target->n);

    if(iftDictContainKey("--f-x", args, NULL) && iftDictContainKey("--f-x", args, NULL) && iftDictContainKey("--f-x", args, NULL)) {
        first_seed_vx.x = iftGetLongValFromDict("--f-x", args);
        first_seed_vx.y = iftGetLongValFromDict("--f-y", args);
        first_seed_vx.z = iftGetLongValFromDict("--f-z", args);

        iftSetCellSearchRegion(FOM, fst->label, search_region, first_seed_vx);

        iftVoxel best_position = iftSegmentSingleCellByWatershedMSPS(target, fst, FOM, first_seed_vx, search_region,
                                                                     NULL);

        fprintf(stderr, "Starting position %d %d %d; final position %d %d %d\n\n", first_seed_vx.x, first_seed_vx.y,
                first_seed_vx.z,
                best_position.x, best_position.y, best_position.z);

        iftWriteImageByExt(regmin, "regmin%s", iftFileExt(img_fname));
        if(!iftIs3DImage(img)) {
            iftImage *tmp = iftCopyImage(target);
            iftAdjRel *B = iftCircular(1.0);
            iftDrawBorders(tmp, fst->label, A, iftRGBtoYCbCr(iftRGBColor(255, 255, 0), 255), B);

            iftWriteImageByExt(tmp, "result.ppm");

            iftDestroyImage(&tmp);
            iftDestroyAdjRel(&B);
        }
    } else {
        iftImage *final_label = iftSegmentCellsByWatershed(target, FOM, A);

        iftDestroyImage(&final_label);
    }

//    iftAdjRel *Abh = (iftIs3DImage(target)) ? iftSpheric(iftMax(1.0, pow(3*cell_size_max/(4*M_PI), 1/3.0))) : iftCircular(sqrt(sqrt(cell_size_max/M_PI)));
//    iftImage *black_hat = iftBlackTopHat(target, Abh);
//    iftWriteImageByExt(black_hat, "black_hat%s", iftFileExt(img_fname));
//    iftDestroyAdjRel(&Abh);
//    iftDestroyImage(&black_hat);

//    iftAdjRel *Awh = (iftIs3DImage(target)) ? iftSpheric(iftMax(1.0, pow(3*cell_size_max/(4*M_PI), 1/3.0))) : iftCircular(sqrt(sqrt(cell_size_max/M_PI)));
//    iftImage *white_hat = iftWhiteTopHat(target, Awh);
//    iftWriteImageByExt(white_hat, "white_hat%s", iftFileExt(img_fname));
//    iftDestroyAdjRel(&Awh);
//    iftDestroyImage(&white_hat);

    iftDestroyImage(&regmin);
    iftDestroyImageForest(&fst);
    if(cells_filtered != cells)
        iftDestroyImageArray(&cells_filtered);

    iftDestroyImageArray(&cells);
    iftDestroyImageArray(&cells_aligned);
    iftDestroyAdjRel(&A);

    iftDestroyCellFOM(&FOM);

    iftDestroyImage(&label);
    iftDestroyImage(&img);
    iftDestroyImage(&target);

    /***********************************/
    MemDinFinal = iftMemoryUsed();
    iftVerifyMemory(MemDinInicial, MemDinFinal);

    return 0;
}


void iftSetCellSearchRegion(  iftCellFOM *FOM,   iftImage *label, iftBMap *search_region, iftVoxel cand_position) {
    iftVoxel model_center = iftImageCenter(FOM->model);

    // Setting only the region under the model image's current position in the target image as search region
    for(int p = 0; p < FOM->model->n; p++) {
        iftVoxel v      = iftFGetVoxelCoord(FOM->model, p);
        iftVoxel v_t = (iftVoxel)iftVectorSum((iftVoxel)iftVectorSub(v, model_center), cand_position);

        if(iftValidVoxel(label, v_t)) {
            int q = iftGetVoxelIndex(label, v_t);

            // We only allow a voxel to be part of the search region if it has not been previously segmented, if label is
            // given
            if(label->val[q] == 0) {
                iftBMapSet1(search_region, q);
            }
        }
    }
}


iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = "This is a program to compute a fuzzy model from an image with cells";

    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input Image"},
            {.short_name = "-l", .long_name = "--label", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Label image"},
            {.short_name = "-t", .long_name = "--test", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Test image"},
            {.short_name = "", .long_name = "--min-vol", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="Minimum cell volume"},
            {.short_name = "", .long_name = "--max-vol", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="Maximum cell volume"},
            {.short_name = "", .long_name = "--f-x", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="First x"},
            {.short_name = "", .long_name = "--f-y", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="First y"},
            {.short_name = "", .long_name = "--f-z", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="First z"}

    };
    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

iftImageArray *iftExtractCells(iftImage *label) {
    int lb, max_lb, i = 0, nlabels = 0;
    iftImageArray *cells = NULL;
    iftBoundingBox *bb;

    max_lb = iftMaximumValue(label);

    bb = iftMinLabelsBoundingBox(label, NULL);

    for(lb = 2; lb <= max_lb; lb++) {
        if(bb[lb].end.x >= 0 && bb[lb].end.y >= 0 && bb[lb].end.z >= 0)
            nlabels++;
    }

    cells = iftCreateImageArray(nlabels);

    for(lb = 2; lb <= max_lb; lb++) {
        if(bb[lb].end.x >= 0 && bb[lb].end.y >= 0 && bb[lb].end.z >= 0)
            cells->images[i++] = iftExtractObjectInsideROI(label, bb[lb], lb);
    }

    iftFree(bb);

    return cells;
}


iftImageArray *iftFilterObjectsByVolume(iftImageArray *images, int vol_min, int vol_max) {
    iftImageArray *images_filtered = NULL;
    iftSet *elems = NULL;
    size_t nobjects = 0;

    for(int i = 0; i < (int)images->n; i++) {
        int lb = iftMaximumValue(images->images[i]);
        int obj_vol = iftCountObjectSpels(images->images[i], lb);

        if(obj_vol >= vol_min && obj_vol <= vol_max) {
            iftInsertSet(&elems, i);
            nobjects++;
        }
    }

    if(nobjects > 0) {
        images_filtered = iftCreateImageArray(nobjects);

        while(elems != NULL) {
            int i = iftRemoveSet(&elems);

            // Saving backwards since iftSet inserts as LIFO. This should be unnecessary in practice.
            images_filtered->images[nobjects - 1] = iftCopyImage(images->images[i]);
            nobjects--;
        }
    }

    return images_filtered;
}


iftImageArray *iftAlignImagesByCentroid(iftImageArray *images, int padding) {
    int xsize, ysize, zsize;
    iftBoundingBox *bbs = NULL;
    iftPoint *gcs = NULL, max_gcs;
    iftImageArray *aligned = NULL;

    bbs = (iftBoundingBox*)iftAlloc(images->n, sizeof(iftBoundingBox));
    gcs = (iftPoint*)iftAlloc(images->n, sizeof(iftPoint));

    max_gcs.x = max_gcs.y = max_gcs.z = IFT_INFINITY_FLT_NEG;
    for(size_t i = 0; i < images->n; i++) {
        bbs[i] = iftMinBoundingBox(images->images[i], &gcs[i]);

        max_gcs.x = iftMax(max_gcs.x, iftMax(gcs[i].x, bbs[i].end.x - gcs[i].x));
        max_gcs.y = iftMax(max_gcs.y, iftMax(gcs[i].y, bbs[i].end.y - gcs[i].y));
        max_gcs.z = iftMax(max_gcs.z, iftMax(gcs[i].z, bbs[i].end.z - gcs[i].z));
    }

    // Choosing an odd size for the target image
    xsize = 2*(padding + 1 + (int)ceilf(max_gcs.x)) + 1;
    ysize = 2*(padding + 1 + (int)ceilf(max_gcs.y)) + 1;
    zsize = 1;

    if(iftIs3DImage(images->images[0]))
        zsize = 2*(padding + 1 + (int)ceilf(max_gcs.z)) + 1;

    aligned = iftCreateImageArray(images->n);

    for(size_t i = 0; i < images->n; i++) {
        aligned->images[i] = iftCreateImage(xsize, ysize, zsize);

        iftInsertROIByPosition(images->images[i], iftPointToVoxel(gcs[i]), aligned->images[i], iftImageCenter(aligned->images[i]));
    }

    iftFree(bbs);
    iftFree(gcs);

    return aligned;
}

iftFImage *iftFuzzyCellModelByAverage(iftImageArray *images) {
    iftFImage *fimg = NULL;

    fimg = iftCreateFImage(images->images[0]->xsize, images->images[0]->ysize, images->images[0]->zsize);

    for(size_t i = 0; i < images->n; i++) {
        for(int p = 0; p < fimg->n; p++) {
            fimg->val[p] += (images->images[i]->val[p] > 0) ? 1 : 0;
        }
    }

    for(int p = 0; p < fimg->n; p++) {
        fimg->val[p] /= iftMax(images->n, 1);
    }

    return fimg;
}


bool iftHasUncertainNeighbor(  iftFImage *model, float bkg_thresh, float obj_thresh,   iftAdjRel *A, int p) {
    bool has_adj = false;
    iftVoxel u,v;

    u = iftFGetVoxelCoord(model, p);
    for(int i = 1; i < A->n; i++) {
        v = iftGetAdjacentVoxel(A, u, i);

        if(iftValidVoxel(model, v)) {
            int q = iftGetVoxelIndex(model, v);

            if (model->val[q] > bkg_thresh && model->val[q] < obj_thresh) {
                has_adj = true;
                break;
            }
        }
    }

    return has_adj;
}

iftCellFOM * iftCreateCellFOM(iftFImage *model, iftAdjRel *A, float bkg_thresh, float obj_thresh) {
    int nseeds_in = 0, nseeds_out = 0, nuncertain = 0;
    iftVoxel center = iftImageCenter(model);
    iftSet *Sin = NULL, *Sout = NULL, *Suncertain = NULL;
    iftAdjRel **ALabels = NULL, *AUncertain = NULL;
    iftIntArray *labels = NULL;
    iftCellFOM *FOM = NULL;

    // Selecting seeds and uncertainty region
    for(int p = 0; p < model->n; p++) {
        if(model->val[p] <= bkg_thresh) {
            bool has_adj = iftHasUncertainNeighbor(model, bkg_thresh, obj_thresh, A, p);

            if(has_adj) {
                iftInsertSet(&Sout, p);
                nseeds_out++;
            }
        } else if(model->val[p] >= obj_thresh) {
            bool has_adj = iftHasUncertainNeighbor(model, bkg_thresh, obj_thresh, A, p);

            if(has_adj) {
                iftInsertSet(&Sin, p);
                nseeds_in++;
            }
        } else {
            iftInsertSet(&Suncertain, p);
            nuncertain++;
        }
    }

    // Allocating adjacency relation
    ALabels = (iftAdjRel**)iftAlloc(2, sizeof(iftAdjRel*));
    AUncertain = iftCreateAdjRel(nuncertain);

    ALabels[0] = iftCreateAdjRel(nseeds_out);
    ALabels[1] = iftCreateAdjRel(nseeds_in);

    labels = iftCreateIntArray(2);

    labels->val[0] = 0;
    labels->val[1] = 1;

    // Copying data
    int i = 0;
    for(iftSet *S = Sin; S != NULL; S = S->next) {
        iftVoxel v = iftFGetVoxelCoord(model, S->elem);

        ALabels[1]->dx[i] = v.x - center.x;
        ALabels[1]->dy[i] = v.y - center.y;
        ALabels[1]->dz[i] = v.z - center.z;

        i++;
    }

    i = 0;
    for(iftSet *S = Sout; S != NULL; S = S->next) {
        iftVoxel v = iftFGetVoxelCoord(model, S->elem);

        ALabels[0]->dx[i] = v.x - center.x;
        ALabels[0]->dy[i] = v.y - center.y;
        ALabels[0]->dz[i] = v.z - center.z;
        i++;
    }

    i = 0;
    for(iftSet *S = Suncertain; S != NULL; S = S->next) {
        iftVoxel v = iftFGetVoxelCoord(model, S->elem);

        AUncertain->dx[i] = v.x - center.x;
        AUncertain->dy[i] = v.y - center.y;
        AUncertain->dz[i] = v.z - center.z;
        i++;
    }

    iftDestroySet(&Sin);
    iftDestroySet(&Sout);
    iftDestroySet(&Suncertain);

    FOM = iftAllocCellFOM(model, ALabels, AUncertain, labels);

    return FOM;
}

iftCellFOM *iftAllocCellFOM(iftFImage *model, iftAdjRel **ALabels, iftAdjRel *AUncertain, iftIntArray *labels) {
    iftCellFOM *FOM = NULL;

    FOM             = (iftCellFOM*)iftAlloc(1, sizeof(iftCellFOM));
    FOM->labels     = labels;
    FOM->ALabels    = ALabels;
    FOM->AUncertain = AUncertain;
    FOM->model      = model;

    return FOM;
}

void iftDestroyCellFOM(iftCellFOM **FOM) {

    if(FOM != NULL && *FOM != NULL) {
        iftDestroyAdjRel(&(*FOM)->AUncertain);

        for(size_t i = 0; i < (*FOM)->labels->n; i++) {
            iftDestroyAdjRel(&(*FOM)->ALabels[i]);
        }

        iftDestroyIntArray(&(*FOM)->labels);
        iftFree((*FOM)->AUncertain);

        iftDestroyFImage(&(*FOM)->model);
        iftFree(*FOM);

        *FOM = NULL;
    }
}

void iftWriteCellFOM(  iftCellFOM *FOM, const char *filename) {
    iftJson *json = iftCreateJsonRoot();
    char *fname = NULL;
    char *dir = iftMakeTempDir("tmp_CELL_FOM");


    fname = iftJoinPathnames(2, dir, "ALabels0.dat");
    iftWriteAdjRel(FOM->ALabels[0], fname);
    iftFree(fname);

    fname = iftJoinPathnames(2, dir, "ALabels1.datr");
    iftWriteAdjRel(FOM->ALabels[1], fname);
    iftFree(fname);

    iftAddIntArrayToJson(json, "labels", FOM->labels);
    fname = iftJoinPathnames(2, dir, "info.json");
    iftWriteJson(json, fname);
    iftFree(fname);

    fname = iftJoinPathnames(2, dir, "mode.fimg");
    iftWriteFImage(FOM->model, fname);
    iftFree(fname);

    iftZipDirContent(dir, filename);

    iftRemoveDir(dir);

    iftFree(dir);
    iftDestroyJson(&json);
}
//
//iftCellFOM* iftReadCellFOM(const char *filename) {
//    iftCellFOM *FOM = NULL;
//    iftJson *json = NULL;
//    char *fname = NULL;
//    char *dir = iftMakeTempDir("tmp_CELL_FOM");
//    iftIntArray *
//
//    iftUnzipFile(filename, dir);
//
//    fname = iftJoinPathnames(2, dir, "ALabels0.dat");
//    iftWriteAdjRel(FOM->ALabels[0], fname);
//    iftFree(fname);
//
//    fname = iftJoinPathnames(2, dir, "ALabels1.datr");
//    iftWriteAdjRel(FOM->ALabels[1], fname);
//    iftFree(fname);
//
//    iftAddIntArrayToJson(json, "labels", FOM->labels);
//    fname = iftJoinPathnames(2, dir, "info.json");
//    json = iftReadJson(fname);
//
//
//    iftWriteJson(json, fname);
//    iftFree(fname);
//
//    fname = iftJoinPathnames(2, dir, "mode.fimg");
//    iftWriteFImage(FOM->model, fname);
//    iftFree(fname);
//
//    iftRemoveDir(dir);
//
//    iftFree(dir);
//    iftDestroyJson(&json);
//}

void iftDrawAdjRel(iftImage *img, iftAdjRel *A, int p, iftColor YCbCr, bool include_center) {
    if(!iftIsColorImage(img))
        iftSetCbCr(img, 128);

    iftVoxel center = iftGetVoxelCoord(img, p);

    for(int i = ((include_center) ? 0 : 1); i < A->n; i++) {
        iftVoxel v = iftGetAdjacentVoxel(A, center, i);

        if(iftValidVoxel(img, v))
            iftSetYCbCr(img, iftGetVoxelIndex(img, v), YCbCr);
    }
}


/***********************************************/


iftVoxel iftSegmentSingleCellByWatershedMSPS(iftImage *orig, iftImageForest *fst, iftCellFOM *FOM, iftVoxel cand_pos,
                                             iftBMap *search_region, iftLabeledSet **objects) {
    // CHECKERS
    if (!iftValidVoxel(fst->img, cand_pos))
        iftError("Object Model Center does not belong to the Test Image Domain\n" \
                 "model center (x, y, z) = (%d, %d, %d)\n",
                 "Test Grad. Image Domain (xsize, ysize, zsize) = (%d, %d, %d)\n",
                 "iftSegmentObjByWatershedMSPS", cand_pos.x, cand_pos.y, cand_pos.z,
                 fst->img->xsize, fst->img->ysize, fst->img->zsize);

    iftOMDelinAlgData *del_data = iftCreateOMDelinAlgWatershedData(orig, fst);
    iftOMRecogAlgData *rec_data = iftCreateOMRecogAlgDataMeanCut(fst);

    // builds the MSPS Problem
    iftObjectModelProb *problem = iftCreateObjectModelProb(FOM, FOM->ALabels, FOM->AUncertain,
                                                           FOM->labels, del_data, rec_data, search_region);
    iftMSPS *msps = iftMSPSObjectModelSearchByTranslationSingleCenterInit(problem, 2, 1, 5);


    // run MSPS
    iftVoxel best_pos;
    iftMSPSObjectModelSearchByTranslationSingleCenterOptimization(msps, cand_pos, &best_pos);

    // delineates the object
    iftResetImageForest(fst);
    iftMSPSDelineateObjectWithModel(problem, &best_pos, objects, NULL);

    // DESTROYERS
    iftDestroyOMDelinAlgWatershedData(&del_data);
    iftDestroyOMRecogDataMeanCut(&rec_data);
    iftDestroyObjectModelProb(&problem);
    iftDestroyMSPS(&msps);

    return best_pos;
}


iftImage *iftWhiteTopHat(iftImage *img, iftAdjRel *A) {
    iftImage *op = NULL, *res = NULL;

    op = iftOpen(img, A);
    res = iftSub(img, op);

    iftDestroyImage(&op);

    return res;
}

iftImage *iftBlackTopHat(iftImage *img, iftAdjRel *A) {
    iftImage *cls = NULL, *res = NULL;

    cls = iftClose(img, A);
    res = iftSub(cls, img);

    iftDestroyImage(&cls);

    return res;
}

void iftSetValueOnAdjRel(iftImage *img, iftAdjRel *A, int p, int value, bool include_center) {
    iftVoxel center = iftGetVoxelCoord(img, p);

    for(int i = ((include_center) ? 0 : 1); i < A->n; i++) {
        iftVoxel v = iftGetAdjacentVoxel(A, center, i);

        if(iftValidVoxel(img, v))
            img->val[iftGetVoxelIndex(img, v)] = value;
    }
}

iftImage *iftSegmentCellsByWatershed(iftImage *orig, iftCellFOM *FOM, iftAdjRel *A) {
    int lb = 1;
    iftImageForest *fst = NULL;
    iftImage *final_seeds = NULL;
    iftImage *final_label = NULL;
    iftImage *regmin = NULL;
    iftBMap *search_region = NULL;

    fst = iftCreateImageForest(orig, A);
    regmin = iftRegionalMinima(orig);
    final_label = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
    final_seeds = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
    search_region = iftCreateBMap(orig->n);

    //    iftSetImage(final_seeds, IFT_NIL);

    for (int p = 0; p < regmin->n; p++) {
        if (regmin->val[p] > 0) {
            iftVoxel cand_pos = iftGetVoxelCoord(regmin, p);
            iftVoxel best_pos;
            iftLabeledSet *objects = NULL;

            FOM->labels->val[1] = lb;

            iftFillBMap(search_region, 0);

            iftSetCellSearchRegion(FOM, final_label, search_region, cand_pos);

            best_pos = iftSegmentSingleCellByWatershedMSPS(orig, fst, FOM, cand_pos, search_region, &objects);

            iftSetValueOnAdjRel(final_seeds, FOM->ALabels[1], iftGetVoxelIndex(regmin, best_pos), lb, true);

            while(objects != NULL) {
                int foo_lb;
                int q = iftRemoveLabeledSet(&objects, &foo_lb);

                if(foo_lb == lb) {
                    // Erasing regional minima located under the current segmentation
                    regmin->val[q] = 0;

                    // Copying segmentation if the voxel has not been segmented yet
                    if(final_label->val[q] == 0)
                        final_label->val[q] = lb;
                }
            }


            if (iftIs3DImage(orig)) {
                iftWriteImageByExt(final_seeds, "seeds_%05d.scn", lb);
                iftWriteImageByExt(final_label, "label_%05d.scn", lb);
            } else {
                iftImage *tmp = iftCopyImage(orig);
                iftAdjRel *B = iftCircular(0.0);
                iftDrawObject(tmp, final_seeds, iftRGBtoYCbCr(iftRGBColor(255, 255, 0), 255), B);
                iftWriteImageByExt(tmp, "seeds_color_%05d.ppm", lb);

                iftDestroyImage(&tmp);

                tmp = iftCopyImage(orig);

                iftDrawBorders(tmp, final_label, A, iftRGBtoYCbCr(iftRGBColor(255, 255, 0), 255), B);
                iftWriteImageByExt(tmp, "label_color_%05d.ppm", lb);

                iftDrawObject(tmp, final_seeds, iftRGBtoYCbCr(iftRGBColor(255, 0, 0), 255), B);

                iftWriteImageByExt(tmp, "label_seeds_color_%05d.ppm", lb);

                iftDestroyImage(&tmp);
                iftDestroyAdjRel(&B);
            }
            lb++;
        }
    }

    iftDestroyImage(&final_label);

    for(int p = 0; p < final_seeds->n; p++) {
        if(final_seeds->val[p] == 0)
            final_seeds->val[p] = IFT_NIL;
    }

    iftLabeledSet *seeds = iftLabeledSetFromSeedImage(final_seeds);
    final_label = iftWatershed(orig, A, seeds, NULL);
    iftDestroyLabeledSet(&seeds);

    if (iftIs3DImage(orig)) {
        iftWriteImageByExt(final_seeds, "final_seeds.scn");
    } else {
        iftImage *tmp = iftCopyImage(orig);
        iftAdjRel *B = iftCircular(0.0);
        for(int p = 0; p < final_seeds->n; p++) {
            if(final_seeds->val[p] < 0)
                final_seeds->val[p] = 0;
        }

        iftDrawObject(tmp, final_seeds, iftRGBtoYCbCr(iftRGBColor(255, 255, 0), 255), B);
        iftWriteImageByExt(tmp, "final_seeds_color.ppm");

        iftDestroyImage(&tmp);

        tmp = iftCopyImage(orig);
        iftDrawBorders(tmp, final_label, A, iftRGBtoYCbCr(iftRGBColor(255, 255, 0), 255), B);
        iftWriteImageByExt(tmp, "final_label_color.ppm");

        iftDestroyImage(&tmp);
        iftDestroyAdjRel(&B);
    }

    // Cleaning up!
    iftDestroyImageForest(&fst);
    iftDestroyImage(&regmin);
    iftDestroyImage(&final_seeds);
    iftDestroyBMap(&search_region);


    return  final_label;
}
