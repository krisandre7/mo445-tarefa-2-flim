#include "iftSeedObjectModel.h"


iftSeedObjectModel *iftCreateSeedObjectModel(iftFImage *model, iftImage *candidate_seed_label, iftVoxel model_center,
                                             iftBoundingBox uncert_region_bbox, iftVector search_start,
                                             iftVector search_end) {
    iftSeedObjectModel *seed_model = NULL;

    seed_model = (iftSeedObjectModel*)calloc(1, sizeof(iftSeedObjectModel));

    seed_model->model = model;
    seed_model->candidate_seed_label = candidate_seed_label;
    seed_model->model_center = model_center;
    seed_model->uncert_region_bbox = uncert_region_bbox;
    seed_model->start = search_start;
    seed_model->end = search_end;

    // These fields should be assigned later
    seed_model->seeds = NULL;
    seed_model->selected_threshold = 0.0;
    seed_model->intensities_mode = 0.0;
    seed_model->intensities_mode_deviation = 0.0;

    return seed_model;
}

void iftDestroySeedObjectModel(iftSeedObjectModel **model) {
    if(model != NULL && *model != NULL) {
        iftDestroyFImage(&(*model)->model);
        iftDestroyImage(&(*model)->candidate_seed_label);
        iftDestroyLabeledSet(&(*model)->seeds);
        free(*model);
        *model = NULL;
    }
}

iftSeedObjectModel *iftCreateSeedObjectModelByAverage(const char *ref_seed_basename, iftFileSet *seed_img_files,
                                                            iftFileSet *label_img_files) {
    int i, lb, p;
    iftPoint gc;
    iftBoundingBox uncert_region_bbox, bbox;
    iftFImage *model_image = NULL;
    iftImage *seed_image = NULL;
    iftImage *label = NULL;
    iftImage **obj_seed_label_freq = NULL;
    iftImage *candidate_seed_label = NULL;
    iftSeedObjectModel *model = NULL;
    iftImage *ref_seed_image = NULL;
    iftImage *ref_label = NULL;
    iftLabeledSet *seeds = NULL;

    iftVector start, end;
    iftVoxel model_center;

    int ref_seed_file_idx;
    int nobjects = 0;

    /* Loading reference image data */
    ref_seed_file_idx = iftFindFileByBasename(seed_img_files, ref_seed_basename, "");

    ref_label = iftReadImageByExt(label_img_files->files[ref_seed_file_idx]->pathname);
    seeds = iftReadSeeds(seed_img_files->files[ref_seed_file_idx]->pathname, ref_label);
    ref_seed_image = iftSeedImageFromLabeledSet(seeds, ref_label);
    iftDestroyLabeledSet(&seeds);

    /* Loading the data for the first label */
    label = iftReadImageByExt(label_img_files->files[0]->pathname);
    seeds = iftReadSeeds(seed_img_files->files[0]->pathname, label);
    seed_image = iftSeedImageFromLabeledSet(seeds, label);
    iftDestroyLabeledSet(&seeds);

    iftVerifyImageDomains(ref_seed_image, seed_image, "iftCreateSeedObjectModelByAverage");

    /* Allocating model data */
    model_image = iftCreateFImage(seed_image->xsize, seed_image->ysize, seed_image->zsize);
    candidate_seed_label = iftCreateImage(seed_image->xsize, seed_image->ysize, seed_image->zsize);

    nobjects = iftMaximumValue(ref_label);
    /* Allocating one image per label to count label frequency (includes the background */
    obj_seed_label_freq = (iftImage**)calloc(nobjects+1, sizeof(iftImage*));

    for(lb = 0; lb < nobjects+1; lb++) {
        obj_seed_label_freq[lb] = iftCreateImage(seed_image->xsize, seed_image->ysize, seed_image->zsize);
    }

    iftFCopyVoxelSizeFromImage(seed_image, model_image);
    iftCopyVoxelSize(seed_image, candidate_seed_label);

    uncert_region_bbox = iftMinBoundingBox(ref_label, &gc);

    /* Assuming the model's center as the geometric center of the reference image*/
    model_center.x = iftRound(gc.x);
    model_center.y = iftRound(gc.y);
    model_center.z = iftRound(gc.z);

    start.x = start.y = start.z = 0;
    end.x = end.y = end.z = 0;

    for(i = 0; i < seed_img_files->nfiles;) {
        if (i > 0) {
            label = iftReadImageByExt(label_img_files->files[i]->pathname);
            seeds = iftReadSeeds(seed_img_files->files[i]->pathname, label);
            seed_image = iftSeedImageFromLabeledSet(seeds, label);
            iftDestroyLabeledSet(&seeds);
        }

        /* Making the seed image range from 0 to nobjects+1 for the computation of the bounding box */
        for(p = 0; p < seed_image->n; p++)
            seed_image->val[p]++;

        bbox = iftMinBoundingBox(seed_image, &gc);

        uncert_region_bbox.begin.x = iftMin(uncert_region_bbox.begin.x, bbox.begin.x);
        uncert_region_bbox.begin.y = iftMin(uncert_region_bbox.begin.y, bbox.begin.y);
        uncert_region_bbox.begin.z = iftMin(uncert_region_bbox.begin.z, bbox.begin.z);

        uncert_region_bbox.end.x = iftMax(uncert_region_bbox.end.x, bbox.end.x);
        uncert_region_bbox.end.y = iftMax(uncert_region_bbox.end.y, bbox.end.y);
        uncert_region_bbox.end.z = iftMax(uncert_region_bbox.end.z, bbox.end.z);

        bbox = iftMinBoundingBox(label, &gc);

        uncert_region_bbox.begin.x = iftMin(uncert_region_bbox.begin.x, bbox.begin.x);
        uncert_region_bbox.begin.y = iftMin(uncert_region_bbox.begin.y, bbox.begin.y);
        uncert_region_bbox.begin.z = iftMin(uncert_region_bbox.begin.z, bbox.begin.z);

        uncert_region_bbox.end.x = iftMax(uncert_region_bbox.end.x, bbox.end.x);
        uncert_region_bbox.end.y = iftMax(uncert_region_bbox.end.y, bbox.end.y);
        uncert_region_bbox.end.z = iftMax(uncert_region_bbox.end.z, bbox.end.z);

        /* Counting the label frequency of the seed image (note that the reference seed image is counted here as well) */
        for (p = 0; p < seed_image->n; p++) {
            int seed = seed_image->val[p]-1;

            if(seed >= 0)
                obj_seed_label_freq[seed]->val[p]++;
        }
        iftDestroyImage(&seed_image);
        iftDestroyImage(&label);

        i++;
    }

    /* Averaging seeds according to their labels, and assigning the label of candidate seeds (voxels with non-zero
     * "probability") by majority vote
     */
    for(p = 0; p < model_image->n; p++) {
        int max_freq_lb = IFT_NIL, max_freq = 0; // We only  count voxels with non-zero frequency, thus, the initial max_freq is 0

        for(lb = 0; lb < nobjects+1; lb++) {
            if(obj_seed_label_freq[lb]->val[p] > max_freq){
                max_freq_lb = lb;
                max_freq = obj_seed_label_freq[lb]->val[p];
            }
        }

        if(max_freq_lb >= 0) {
            model_image->val[p] = max_freq / (double)iftMax(1.0, seed_img_files->nfiles);
            candidate_seed_label->val[p] = max_freq_lb+1; // Background will be labeled as 1 and foreground starts from 2
                                                          // to prevent images from being saved with negative values to disk
        }
    }


    for(lb = 0; lb < nobjects+1; lb++) {
        iftDestroyImage(&obj_seed_label_freq[lb]);
    }

    iftDestroyImage(&ref_seed_image);
    iftDestroyImage(&ref_label);

    /* Creating model*/
    model = iftCreateSeedObjectModel(model_image, candidate_seed_label, model_center, uncert_region_bbox, start, end);

    fprintf(stderr,"Maximum membership %f\n", iftFMaximumValue(model->model));
    fprintf(stderr,"Minimum membership %f\n", iftFMinimumValue(model->model));

    return model;
}


/**
 * @brief This function determines, for each label, the maximum threshold that guarantees that the label will still
 * be represented by seeds and then selects the minimum among all thresholds as the maximum threshold that should be
 * chosen.
 *
 * @author Thiago Vallin Spina
 * @date Jan 27, 2016
 *
 * @param model The seed object model.
 * @return The maximum representative threshold.
 */
double iftDetermineMaximumRepresentativeThresholdForSeedObjectModel(iftSeedObjectModel *model) {
    int p, lb, nlabels;
    double min_thresh = IFT_INFINITY_DBL;
    double *max_label_threshold = NULL;

    nlabels = iftMaximumValue(model->candidate_seed_label);

    max_label_threshold = iftAllocDoubleArray(nlabels);

    for(lb = 0; lb < nlabels; lb++) {
        max_label_threshold[lb] = IFT_INFINITY_DBL_NEG;
    }

    /* Computing the maximum threshold supported for each label in order for it to be represented by seeds in the model */
    for(p = 0; p < model->candidate_seed_label->n; p++) {
        if(model->candidate_seed_label->val[p] > 0) {
            lb = model->candidate_seed_label->val[p]-1; // Recall that the background candidate seed label is 1, the first object is 2,...

            max_label_threshold[lb] = iftMax(max_label_threshold[lb], model->model->val[p]);
        }
    }

    // Selecting the minimum representative threshold among all
    min_thresh = max_label_threshold[iftDArgmin(max_label_threshold, nlabels)];

    free(max_label_threshold);

    return min_thresh;

}

void iftDetermineSeedsForSeedObjectModel(iftSeedObjectModel *model, double thresh) {
    double min_supported_thresh = IFT_INFINITY_DBL;
    iftImage *mask = NULL;
    iftImage *seeds_image = NULL;
    iftLabeledSet *aux = NULL;
    iftImage *relabeled_components = NULL;
    iftAdjRel *A = NULL;

    min_supported_thresh = iftDetermineMaximumRepresentativeThresholdForSeedObjectModel(model);

    if(thresh > min_supported_thresh) {
        iftWarning("The threshold is above the minimum necessary for selecting seeds for all of the labels in the model.\n"
                    "Decreasing threshold from %.4lf to %.4lf.\n", "iftDetermineSeedsForSeedObjectModel", thresh, min_supported_thresh);

        thresh = min_supported_thresh;
    }

    model->selected_threshold = thresh;
    mask = iftFThreshold(model->model, thresh, 1.0, 1);

    seeds_image = iftMask(model->candidate_seed_label, mask);

    if(iftIs3DImage(model->candidate_seed_label))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.5);

    /* Relabeling seed components to determine marker ids*/
    relabeled_components = iftRelabelRegions(seeds_image, A);


    /* IMPORTANT NOTE: iftAddValue does not allow the image to have negative values.. so we do it manually
     * here for labels to be assigned correctly to the seeds
     */
    for(int i = 0; i < seeds_image->n; i++)
        seeds_image->val[i]--;

    iftDestroyLabeledSet(&model->seeds);
    model->seeds = iftLabeledSetFromSeedImage(seeds_image);

    for(aux = model->seeds; aux != NULL; aux = aux->next) {
        aux->marker = relabeled_components->val[aux->elem];
    }
    iftDestroyImage(&mask);
    iftDestroyImage(&seeds_image);
    iftDestroyImage(&relabeled_components);
    iftDestroyAdjRel(&A);
}

void iftDetermineTextureInformationForSeedObjectModel(iftSeedObjectModel *model,   iftFileSet *orig_img_files,
                                                        iftFileSet *label_img_files, int normalization_value) {
    iftImageSetVoxelIntensityStatistics(orig_img_files, label_img_files, &model->intensities_mode,
                                        &model->intensities_mode_deviation, normalization_value); // Normalizing all images to
}

void iftWriteSeedObjectModel(iftSeedObjectModel *model, const char *output_basename) {
    char *model_image_file = NULL;
    char *model_file = NULL;
    char *model_cand_seed_file = NULL;
    char *model_seeds_image_file = NULL;
    char *model_seeds_file = NULL;
    char *model_json_file = NULL;
    char *base_dir = NULL;
    char ext_with_dot[10];
    iftImage *model_image = NULL;
    iftImage *seeds_image = NULL;
    iftJson *json = NULL;
    iftIntArray *uncert_region_bbox_arr = NULL;
    iftIntArray *model_center_arr = NULL;
    iftDblArray *start_end_arr = NULL;

    /* Creating base directory if necessary */
    base_dir = iftParentDir(output_basename);

    if(!iftCompareStrings(base_dir, ""))
        iftMakeDir(base_dir);
    free(base_dir);

    if(iftIs3DImage(model->candidate_seed_label))
        sprintf(ext_with_dot, ".scn");
    else
        sprintf(ext_with_dot, ".pgm");

    /* Preparing strings */
    model_file = iftConcatStrings(2, output_basename, ".fimg");
    model_image_file = iftConcatStrings(3, output_basename, "_image", ext_with_dot);
    model_cand_seed_file = iftConcatStrings(3, output_basename, "_candidate_seed_label", ext_with_dot);
    model_seeds_image_file = iftConcatStrings(3, output_basename, "_seeds", ext_with_dot);
    model_seeds_file = iftConcatStrings(2, output_basename, "_seeds.txt");
    model_json_file = iftConcatStrings(2, output_basename, "_info.json");

    if(iftIs3DFImage(model->model)) {
        model_image = iftFImageToImage(model->model, 4095);
    } else {
        model_image = iftFImageToImage(model->model, UCHAR_MAX);
    }
    seeds_image = iftSeedImageFromLabeledSet(model->seeds, model_image);

    json = iftCreateJsonRoot();

    model_center_arr = iftCreateIntArray(3);
    uncert_region_bbox_arr = iftCreateIntArray(6);
    start_end_arr = iftCreateDblArray(6);

    uncert_region_bbox_arr->val[0] = model->uncert_region_bbox.begin.x;
    uncert_region_bbox_arr->val[1] = model->uncert_region_bbox.begin.y;
    uncert_region_bbox_arr->val[2] = model->uncert_region_bbox.begin.z;
    uncert_region_bbox_arr->val[3] = model->uncert_region_bbox.end.x;
    uncert_region_bbox_arr->val[4] = model->uncert_region_bbox.end.y;
    uncert_region_bbox_arr->val[5] = model->uncert_region_bbox.end.z;

    start_end_arr->val[0] = model->start.x;
    start_end_arr->val[1] = model->start.y;
    start_end_arr->val[2] = model->start.z;
    start_end_arr->val[3] = model->end.x;
    start_end_arr->val[4] = model->end.y;
    start_end_arr->val[5] = model->end.z;

    model_center_arr->val[0] = model->model_center.x;
    model_center_arr->val[1] = model->model_center.y;
    model_center_arr->val[2] = model->model_center.z;

    iftAddStringToJson(json, "model_extension", ext_with_dot);
    iftAddIntArrayToJson(json, "model_center", model_center_arr);
    iftAddIntArrayToJson(json, "uncert_region_bbox", uncert_region_bbox_arr);
    iftAddDoubleArrayToJson(json, "start_end", start_end_arr);
    iftAddDoubleToJson(json, "selected_threshold", model->selected_threshold);
    iftAddDoubleToJson(json, "intensities_mode", model->intensities_mode);
    iftAddDoubleToJson(json, "intensities_mode_deviation", model->intensities_mode_deviation);

    /* Writing model data */
    iftWriteFImage(model->model, model_file);
    iftWriteImageByExt(model_image, model_image_file);
    iftWriteImageByExt(model->candidate_seed_label, model_cand_seed_file);
    iftWriteImageByExt(seeds_image, model_seeds_image_file);
    iftWriteSeeds(model_seeds_file, model->seeds, seeds_image);
    iftWriteJson(json, model_json_file);

    /* Cleaning up */
    free(model_file);
    free(model_image_file);
    free(model_cand_seed_file);
    free(model_seeds_image_file);
    free(model_seeds_file);
    free(model_json_file);
    iftDestroyImage(&model_image);
    iftDestroyImage(&seeds_image);
    iftDestroyIntArray(&uncert_region_bbox_arr);
    iftDestroyIntArray(&model_center_arr);
    iftDestroyDblArray(&start_end_arr);
    iftDestroyJson(&json);
}


void iftDrawSeeds(iftImage *img, iftImage *seed_image, iftColorTable *cmap)
{
    int p;

    iftVerifyImageDomains(img, seed_image,"iftDrawLabels");

    if (img->Cb==NULL)
        iftSetCbCr(img,128);

    for (p=0; p < img->n; p++) {
        iftColor color_lb, color;

        if(seed_image->val[p] >= 0) {
            color_lb = cmap->color[seed_image->val[p]];
            color_lb = iftYCbCrtoRGB(color_lb, 255);

            color = iftGetRGB(img, p, 255);
            color.val[0] = color_lb.val[0];
            color.val[1] = color_lb.val[1];
            color.val[2] = color_lb.val[2];

            iftSetRGB(img, p, color.val[0], color.val[1], color.val[2], 255);
        }
    }
}

iftSeedObjectModel* iftReadSeedObjectModel(char *input_basename) {
    char *model_file = NULL;
    char *model_cand_seed_file = NULL;
    char *model_seeds_file = NULL;
    char *model_json_file = NULL;
    char *ext_with_dot = NULL;
    double selected_threshold;
    iftImage *candidate_seed_label = NULL;
    iftJson *json = NULL;
    iftIntArray *uncert_region_bbox_arr = NULL;
    iftIntArray *model_center_arr = NULL;
    iftDblArray *start_end_arr = NULL;
    iftSeedObjectModel *model = NULL;
    iftFImage *model_prob = NULL;
    iftLabeledSet *seeds = NULL;
    iftBoundingBox uncert_region_bbox;
    iftVector start, end;
    iftVoxel model_center;

    /* Preparing strings */
    model_json_file = iftConcatStrings(2, input_basename, "_info.json");

    if(!iftFileExists(model_json_file))
        iftError("Unable to open seed object model from file %s. It does not exist!\n", input_basename);

    json = iftReadJson(model_json_file);


    ext_with_dot = iftGetJString(json, "model_extension");
    model_file = iftConcatStrings(2, input_basename, ".fimg");
    model_cand_seed_file = iftConcatStrings(3, input_basename, "_candidate_seed_label", ext_with_dot);
    model_seeds_file = iftConcatStrings(2, input_basename, "_seeds.txt");

    model_center_arr = iftGetJIntArray(json, "model_center");
    uncert_region_bbox_arr = iftGetJIntArray(json, "uncert_region_bbox");
    start_end_arr = iftGetJDoubleArray(json, "start_end");
    selected_threshold = iftGetJDouble(json, "selected_threshold");

    model_center.x = model_center_arr->val[0];
    model_center.y = model_center_arr->val[1];
    model_center.z = model_center_arr->val[2];

    uncert_region_bbox.begin.x = uncert_region_bbox_arr->val[0];
    uncert_region_bbox.begin.y = uncert_region_bbox_arr->val[1];
    uncert_region_bbox.begin.z = uncert_region_bbox_arr->val[2];
    uncert_region_bbox.end.x = uncert_region_bbox_arr->val[3];
    uncert_region_bbox.end.y = uncert_region_bbox_arr->val[4];
    uncert_region_bbox.end.z = uncert_region_bbox_arr->val[5];

    start.x = start_end_arr->val[0];
    start.y = start_end_arr->val[1];
    start.z = start_end_arr->val[2];
    end.x = start_end_arr->val[3];
    end.y = start_end_arr->val[4];
    end.z = start_end_arr->val[5];

    /* Reading model data */
    model_prob = iftReadFImage(model_file);
    candidate_seed_label = iftReadImageByExt(model_cand_seed_file);
    seeds = iftReadSeeds(model_seeds_file, candidate_seed_label);

    model = iftCreateSeedObjectModel(model_prob, candidate_seed_label, model_center, uncert_region_bbox, start, end);
    model->seeds = seeds;
    model->selected_threshold = selected_threshold;

    model->intensities_mode = iftGetJDouble(json, "intensities_mode");
    model->intensities_mode_deviation = iftGetJDouble(json, "intensities_mode_deviation");

    /* Cleaning up */
    free(model_file);
    free(model_cand_seed_file);
    free(model_seeds_file);
    free(model_json_file);
    free(ext_with_dot);

    iftDestroyIntArray(&model_center_arr);
    iftDestroyIntArray(&uncert_region_bbox_arr);
    iftDestroyDblArray(&start_end_arr);
    iftDestroyJson(&json);

    return model;
}

iftObjectModelProb *iftSeedObjectModelToObjectModelProb(iftSeedObjectModel *model, int obj_label, iftImage *test_image,
                                                        iftJson *segmentation_config_json) {
    int i, ii, io, p, padding = 2, nin = 0, nout = 0, nuncertain = 0, *label_sizes = NULL, nlabels;
    double gradient_combination_alpha = 0.5;
    char *gradient_type = NULL;
    iftIntArray *labels = NULL;

    iftLabeledSet *aux = NULL;
    iftBMap *used_seed = NULL;
    iftBMap *search_region = NULL;
    iftAdjRel **ALabels = NULL, *AUncertain = NULL;
    iftObjectModelProb *prob = NULL;
    iftVoxel u, start, end;
    iftOMDelineationAlgorithmData *delineation_data = NULL;
    iftOMRecognitionAlgorithmData *recognition_data = NULL;
    iftImageForest *fst = NULL;
    iftImage *basins = NULL;
    iftAdjRel *A = NULL;
    iftSet *existing_labels = NULL;

    gradient_type = iftGetJString(segmentation_config_json, "gradient_type");
    gradient_combination_alpha = iftGetJDouble(segmentation_config_json, "gradient_combination_alpha");

    iftVerifyImageDomains(test_image, model->candidate_seed_label, "iftSeedObjectModelToObjectModelProb");

    search_region = iftCreateBMap(test_image->n);

    start = iftVectorToVoxel(model->start, model->model_center);
    end = iftVectorToVoxel(model->end, model->model_center);

    // Determining search region from bounding box
    for(u.z = start.z; u.z <= end.z; u.z++) {
        for (u.y = start.y; u.y <= end.y; u.y++) {
            for (u.x = start.x; u.x <= end.x; u.x++) {
                p = iftGetVoxelIndex(test_image, u);
                iftBMapSet1(search_region, p);
            }
        }
    }

    if(iftIs3DImage(test_image)) {
        A = iftSpheric(1.0);
    } else {
        A = iftCircular(1.5);
    }

    if(iftCompareStrings(gradient_type, "IMAGE_BASINS"))
        basins = iftImageBasins(test_image, A);
    else if(iftCompareStrings(gradient_type, "IMAGE_GRADIENT_MAGNITUDE"))
        basins = iftImageGradientMagnitude(test_image, A);
    else if(iftCompareStrings(gradient_type, "RESP_SYSTEM_SPECIFIC"))
        basins = iftRespSystemGradient(test_image, A);
    else if(iftCompareStrings(gradient_type, "GAUSSIAN_ENHANCED"))
        basins = iftGaussianEnhancedImageGradient(test_image, A, model->intensities_mode, model->intensities_mode_deviation, gradient_combination_alpha);
    else
        iftError("Invalid gradient type. Please choose among IMAGE_BASINS, IMAGE_GRADIENT_MAGNITUDE, RESP_SYSTEM_SPECIFIC, or GAUSSIAN_ENHANCED", "main");


    fst = iftCreateImageForest(basins, A);
    delineation_data = iftCreateOMDelineationAlgorithmDataWatershed(fst);
    recognition_data = iftCreateOMRecognitionAlgorithmDataWatershedMeanCut(fst);

    // Computing the size of the adjacency relations
    int xsize = (model->uncert_region_bbox.end.x - model->uncert_region_bbox.begin.x + 1 + padding);
    int ysize = (model->uncert_region_bbox.end.y - model->uncert_region_bbox.begin.y + 1 + padding);
    int zsize = (model->uncert_region_bbox.end.z - model->uncert_region_bbox.begin.z + 1 + padding);

    nuncertain = xsize*ysize*zsize;

    nlabels = 0;
    for(aux = model->seeds; aux != NULL; aux = aux->next) {
        if(!iftSetHasElement(existing_labels, aux->label)) {
            iftInsertSet(&existing_labels, aux->label);
            nlabels++;
        }
    }

    labels  = iftCreateIntArray(nlabels);
    label_sizes = iftAllocIntArray(nlabels);

    for(i = 0; i < labels->n; i++) {
        labels->val[i] = IFT_NIL;
    }

    ii = 0;
    for(aux = model->seeds; aux != NULL; aux = aux->next) {
        u = iftGetVoxelCoord(model->candidate_seed_label, aux->elem);

        i = iftFindIntArrayElementIndex(labels->val, labels->n, aux->label);

        if(i < 0) {
            labels->val[ii] = aux->label;
            label_sizes[ii]++;
            ii++;
        } else {
            label_sizes[i]++;
        }

        if(u.x >= model->uncert_region_bbox.begin.x && u.x <= model->uncert_region_bbox.end.x &&
           u.y >= model->uncert_region_bbox.begin.y && u.y <= model->uncert_region_bbox.end.y &&
           u.z >= model->uncert_region_bbox.begin.z && u.z <= model->uncert_region_bbox.end.z) {
            nuncertain--;
        }
    }

    if(nin <= 0 || nout <= 0 || nuncertain <= 0) {
        iftError("No seed available for computing an object model search problem (Number of in: %d out: %d uncertain: %d!!",
                 "iftSeedObjectModelToObjectModelProb");
    }

    // Allocating the adjacency relations
    ALabels = (iftAdjRel**)calloc(nlabels, sizeof(iftAdjRel*));

    ALabels[0] = iftCreateAdjRel(nout);
    ALabels[1] = iftCreateAdjRel(nin);
    AUncertain = iftCreateAdjRel(nuncertain);

    used_seed = iftCreateBMap(model->candidate_seed_label->n);

    // Assigning foreground and background seed displacements
    ii = 0;
    io = 0;
    for(aux = model->seeds; aux != NULL; aux = aux->next) {
        u = iftGetVoxelCoord(model->candidate_seed_label, aux->elem);

        if(aux->label == obj_label) {
            ALabels[1]->dx[ii] = u.x - model->model_center.x;
            ALabels[1]->dy[ii] = u.y - model->model_center.y;
            ALabels[1]->dz[ii] = u.z - model->model_center.z;

            ii++;
        } else {
            ALabels[0]->dx[io] = u.x - model->model_center.x;
            ALabels[0]->dy[io] = u.y - model->model_center.y;
            ALabels[0]->dz[io] = u.z - model->model_center.z;

            io++;
        }


        iftBMapSet1(used_seed, aux->elem);
    }


    // Assigning uncertainty region displacements
    ii = 0;
    for(u.z = model->uncert_region_bbox.begin.z; u.z <= model->uncert_region_bbox.end.z; u.z++) {
        for (u.y = model->uncert_region_bbox.begin.y; u.y <= model->uncert_region_bbox.end.y; u.y++) {
            for (u.x = model->uncert_region_bbox.begin.x; u.x <= model->uncert_region_bbox.end.x; u.x++) {
                p = iftGetVoxelIndex(model->candidate_seed_label, u);

                if(!iftBMapValue(used_seed, p)) {
                    AUncertain->dx[ii] = u.x - model->model_center.x;
                    AUncertain->dy[ii] = u.y - model->model_center.y;
                    AUncertain->dz[ii] = u.z - model->model_center.z;

                    ii++;
                }
            }
        }
    }

    prob = iftCreateObjectModelProb(model, ALabels, AUncertain, labels, delineation_data, recognition_data,
                                    search_region);

    // Cleaning up
    iftDestroyBMap(&used_seed);
    free(gradient_type);
    free(label_sizes);

    return prob;
}

char *iftLongestCommonSuffix(const char *str0, const char *str1) {
    int i, n, n0, n1, max_idx;
    char *suffix = NULL;

    if(str0 != NULL && str1 != NULL) {
        n0 = strlen(str0);
        n1 = strlen(str1);

        n = iftMin(n0, n1);

        for (i = 0, max_idx = n0 - i; i < n && str0[n0 - i] == str1[n1 - i]; i++) {
            max_idx = n0 - i;
        }

        suffix = iftCopyString(str0 + max_idx);
    }

    return suffix;
}

char *iftLongestCommonPrefix(const char *str0, const char *str1) {
    int i, n, n0, n1;
    char *suffix = NULL;

    if(str0 != NULL && str1 != NULL) {
        n0 = strlen(str0);
        n1 = strlen(str1);

        n = iftMin(n0, n1);

        for (i = 0; i < n && str0[i] == str1[i]; i++);

        suffix = iftAllocCharArray(i+1);
        strncpy(suffix, str0, i);
    }

    return suffix;
}


iftImage *iftRespSystemGradient(iftImage *img, iftAdjRel *A)
{
    iftImage *aux, *grad;

    /* Enhance dark regions of the respiratory system: good to assign
       arc weights for interactive lung and bronchi segmentation. */

    aux = iftCreateImage(img->xsize,img->ysize,img->zsize);
    for (int z=0; z < img->zsize; z++) {
        iftImage *slice = iftGetXYSlice(img,z);
        iftImage *cbas  = iftCloseBasins(slice,A,NULL);
        iftImage *res   = iftSub(cbas,slice);
        iftPutXYSlice(aux,res,z);
        iftDestroyImage(&res);
        iftDestroyImage(&cbas);
        iftDestroyImage(&slice);
    }

    grad = iftImageBasins(aux, A);

    iftDestroyImage(&aux);

    return grad;
}
