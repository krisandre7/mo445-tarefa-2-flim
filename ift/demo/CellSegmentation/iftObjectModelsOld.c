#include "iftObjectModels.h"


/********************** PRIVATE FUNCTIONS *************************/
iftObjModelPlus *_iftCreateObjModelPlus() {
    iftObjModelPlus *plus = (iftObjModelPlus*) iftAlloc(1, sizeof(iftObjModelPlus));
    plus->clf_alg         = IFT_NONE;
    return plus;
}


void _iftDestroyObjModelPlus(iftObjModelPlus **plus) {
    if (plus != NULL) {
        iftObjModelPlus *aux = *plus;

        if (aux->clf != NULL) {
            if (aux->clf_alg == IFT_OPF) {
                iftCplGraph *opf_clf = (iftCplGraph*) aux->clf;
                iftDestroyCplGraph(&opf_clf);
            }
        }
    }
}


void _iftWriteObjModelPlus(  iftObjModelPlus *plus, const char *obj_key, iftJson *info, const char *plus_dir) {
    if (plus != NULL) {
        char key[64];
        sprintf(key, "plus:%s", obj_key);
        iftAddJDictReferenceToJson(info, key, iftCreateJDict());
        sprintf(key, "plus:%s:classification-alg", obj_key);



        char *obj_plus_dir = iftJoinPathnames(2, plus_dir, obj_key);
        iftMakeDir(obj_plus_dir);


        char *clf_path = iftJoinPathnames(2, obj_plus_dir, "clf.zip");
        if (plus->clf_alg == IFT_NONE) {
            iftAddStringToJson(info, key, "NONE");
        }
        else if (plus->clf_alg == IFT_OPF) {
            iftAddStringToJson(info, key, "OPF");

            iftWriteCplGraph(plus->clf, clf_path);
            iftFree(clf_path);
        }
        iftFree(obj_plus_dir);

        // smooth factor
        sprintf(key, "plus:%s:smooth-factor", obj_key);
        iftAddDoubleToJson(info, key, plus->smooth_factor);
    }
}


iftObjModelPlus *_iftReadObjModelPlus(iftJson *info, const char *obj_key, const char *plus_dir) {
    iftObjModelPlus *plus = _iftCreateObjModelPlus();

    char key[64];
    sprintf(key, "plus:%s:classification-alg", obj_key);
    const char *clf_alg = iftGetJString(info, key);

    char *obj_plus_dir = iftJoinPathnames(2, plus_dir, obj_key);
    if (!iftDirExists(obj_plus_dir)) {
        iftError("Plus Directory \"%s\" does not exist", "_iftReadObjModelPlus", obj_plus_dir);
    }

    // reads the classifier
    char *clf_path = iftJoinPathnames(2, obj_plus_dir, "clf.zip");
    if (iftCompareStrings(clf_alg, "OPF")) {
        plus->clf_alg = IFT_OPF;
        plus->clf     = iftReadCplGraph(clf_path);
    }

    // smooth factor
    sprintf(key, "plus:%s:smooth-factor", obj_key);
    plus->smooth_factor = iftGetJDouble(info, key);

    // DESTROYERS
    if (obj_plus_dir != NULL)
        iftFree(obj_plus_dir);

    return plus;
}


iftImage *_iftCombineGradImages(iftImage *grad_img1, iftImage *grad_img2, float alpha, int img_range) {
    if (iftAlmostZero(alpha)) {
        return iftCopyImage(grad_img2);
    }
    else if (iftAlmostZero(1.0 - alpha)) {
        return iftCopyImage(grad_img1);
    }
    else {
        iftImage *norm_grad_img1 = iftLinearStretch(grad_img1, iftMinimumValue(grad_img1),
                                                    iftMaximumValue(grad_img1), 0, img_range);
        iftImage *norm_grad_img2 = iftLinearStretch(grad_img2, iftMinimumValue(grad_img2),
                                                    iftMaximumValue(grad_img2), 0, img_range);

        iftImage *combined_grad_img = iftLinearCombination(norm_grad_img1, norm_grad_img2, alpha);


        iftDestroyImage(&norm_grad_img1);
        iftDestroyImage(&norm_grad_img2);

        return combined_grad_img;
    }
}


iftFImage *_iftExtractSingleObjModel(  iftFImage *model,   iftImage *label_map, int target_obj_label) {
    iftFImage *single_model = iftCreateFImage(model->xsize, model->ysize, model->zsize);

#pragma omp parallel for schedule(auto)
    for (int p = 0; p < model->n; p++) {
        if (label_map->val[p] == target_obj_label) {
            single_model->val[p] = model->val[p];
        }
    }

    return single_model;
}


iftImage *_iftDelineateBySOSM(  iftImage *orient_img,   iftImage *grad_img,
                                iftFImage *model,   iftImage *label_map, float e_radius, float d_radius) {
    iftSet *forbidden    = NULL;
    iftLabeledSet *seeds = iftFindModelSeeds(model, label_map, e_radius, d_radius, &forbidden);

    // char *temp = iftMakeTempPathname("tmp/markers_", ".txt");
    // iftWriteSeeds(temp, seeds, grad_img);

    iftImage *seg_img = iftOrientedWatershed(orient_img, grad_img, NULL, seeds, 1, forbidden);
    // iftImage *seg_img = iftWatershed(grad_img, NULL, seeds, forbidden);

    iftDestroySet(&forbidden);
    iftDestroyLabeledSet(&seeds);

    return seg_img;
}


iftImage *_iftDelineateBySOSM_A(  iftImage *orient_img,   iftImage *grad_img,
                                  iftFImage *model,   iftImage *label_map,
                                  iftImage *membership_map, float e_radius, float d_radius) {
    iftSet *forbidden    = NULL;
    iftLabeledSet *seeds = iftFindFilteredModelSeeds(model, label_map, membership_map, e_radius, d_radius,
                                                     &forbidden);

    // char *temp = iftMakeTempPathname("markers_", ".txt");
    // iftWriteSeeds(temp, seeds, grad_img);
    // iftFree(temp);

    iftImage *seg_img = iftOrientedWatershed(orient_img, grad_img, NULL, seeds, 1, forbidden);
    // iftImage *seg_img = iftWatershed(grad_img, NULL, seeds, forbidden);


    iftDestroySet(&forbidden);
    iftDestroyLabeledSet(&seeds);

    return seg_img;
}


iftImage *_iftClassifyImage(  iftImage *test_img,   iftObjModelPlus *plus, int img_max_range,
                            iftImage **fuzzy_map) {
    if (plus->clf == NULL)
        iftError("Texture Classifier is NULL", "_iftClassifyImage");

    iftImage *membership_map = NULL;

    if (plus->clf_alg == IFT_OPF) {
        membership_map = iftClassifyImageByOPF(test_img, plus->clf, plus->smooth_factor,
                                               img_max_range, fuzzy_map);
    }
    else {
        iftError("Classifier not Supported yet!", "_iftClassifyImage");
    }

    // iftWriteImageByExt(membership_map, "membership_map.scn");
    // iftWriteImageByExt(*fuzzy_map, "fuzzy_map.scn");

    return membership_map;
}


iftImage *_iftComputeFinalGradient(  iftImage *test_img,   iftImage *fuzzy_map,   iftFImage *model,
                                   iftGradientAlg grad_alg, float alpha, float beta, int img_max_range) {
    if (grad_alg == IFT_NONE_GRAD) {
        return iftCopyImage(test_img);
    }
    else {
        ///// Combine Test Grad with Fuzzy Map /////
        iftImage *test_grad_img  = iftComputeGradient(test_img, grad_alg);
        iftImage *fuzzy_grad_img = iftComputeGradient(fuzzy_map, grad_alg);

        iftImage *grad_aux = _iftCombineGradImages(test_grad_img, fuzzy_grad_img, alpha, img_max_range);
        iftWriteImageByExt(test_grad_img, "tmp/00_test_grad_img.scn");
        iftWriteImageByExt(fuzzy_grad_img, "tmp/01_fuzzy_grad_img.scn");
        iftWriteImageByExt(grad_aux, "tmp/02_grad_aux.scn");

        iftDestroyImage(&test_grad_img);
        iftDestroyImage(&fuzzy_grad_img);
        ////////////////////////////////////////////

        ///// Combine Grad aux with Object Model Grad /////
        iftImage *model_img      = iftFImageToImage(model, img_max_range);
        iftImage *model_grad_img = iftComputeGradient(model_img, grad_alg);
        iftDestroyImage(&model_img);

        iftImage *final_grad_img = _iftCombineGradImages(grad_aux, model_grad_img, beta, img_max_range);
        iftWriteImageByExt(model_grad_img, "tmp/04_model_grad_img.scn");
        iftWriteImageByExt(final_grad_img, "tmp/05_final_grad_img.scn");

        iftDestroyImage(&grad_aux);
        iftDestroyImage(&model_grad_img);
        ///////////////////////////////////////////////////


        return final_grad_img;
    }
}


iftImage *_iftRunObjModelPlus(  iftImage *test_img,   iftFImage *model,   iftImage *label_map,
                                iftObjModelPlus *plus, iftGradientAlg grad_alg, float alpha,
                              float beta, float e_radius, float d_radius, int img_max_range) {

    ////// Classification ///////
    iftImage *membership_map = NULL;
    iftImage *fuzzy_map      = NULL;
    membership_map = _iftClassifyImage(test_img, plus, img_max_range, &fuzzy_map);

    /////// COMPUTING FINAL GRADIENT ///////
    iftImage *final_grad_img = _iftComputeFinalGradient(test_img, fuzzy_map, model,
                                                        grad_alg, alpha, beta, img_max_range);
    char *tmp = iftMakeTempFile("tmp/grad_", ".scn");
    iftWriteImageByExt(final_grad_img, tmp);

    ///// DELINEATE BY SOSM-A /////
    iftImage *seg_img = _iftDelineateBySOSM_A(membership_map, final_grad_img, model, label_map,
                                              membership_map, e_radius, d_radius);

    // DESTROYERS
    iftDestroyImage(&membership_map);
    iftDestroyImage(&fuzzy_map);
    iftDestroyImage(&final_grad_img);

    return seg_img;
}


iftObjModelPlus *_iftTrainObjModelPlusByOPF(const char *marker_path,   iftImage *ref_img, float smooth_factor) {
    iftLabeledSet *train_markers = iftReadSeeds(marker_path, ref_img);

    iftObjModelPlus *plus = (iftObjModelPlus*) iftAlloc(1, sizeof(iftObjModelPlus));
    plus->smooth_factor = smooth_factor;
    plus->clf           = iftTrainImageClassifierByOPF(ref_img, train_markers, smooth_factor);
    plus->clf_alg       = IFT_OPF;

    iftDestroyLabeledSet(&train_markers);

    return plus;
}


/**
 * Tells if the Object Model has the segmentation Policy ONE_BY_ONE, and has a single object model plus
 * for all objects
 */
bool _iftIsOneByOneWithSingleObjectModelPlus(  iftObjModel *obj_model) {
    return ((obj_model->seg_pol == IFT_ONE_BY_ONE) && (obj_model->n_plus == 1) &&
            (obj_model->labels->n > 1));
}
/******************************************************************/

/********************** PUBLIC FUNCTIONS *************************/
iftObjModel *iftCreateObjModel() {
    iftObjModel *obj_model = (iftObjModel*) iftAlloc(1, sizeof(iftObjModel));
    obj_model->ref_img     = NULL;
    obj_model->plus        = NULL;
    obj_model->seg_pol     = IFT_ONE_BY_ONE;
    obj_model->labels      = NULL;
    obj_model->img_depth   = 0;

    return obj_model;
}


void iftDestroyObjModel(iftObjModel **obj_model) {
    if (obj_model != NULL) {
        iftObjModel *aux = *obj_model;

        if (aux != NULL) {
            if (aux->model != NULL)
                iftDestroyFImage(&aux->model);
            if (aux->label_map != NULL)
                iftDestroyImage(&aux->label_map);
            if (aux->ref_img != NULL)
                iftDestroyImage(&aux->ref_img);

            if (aux->labels != NULL) {
                iftDestroyIntArray(&aux->labels);
            }

            if (aux->plus != NULL) {
                if (aux->seg_pol == IFT_ONE_BY_ONE) {
                    for (int o = 0; o < aux->n_plus; o++) {
                        _iftDestroyObjModelPlus(&aux->plus[o]);
                    }
                }
                else if (aux->seg_pol == IFT_ALL_AT_ONCE) {
                    _iftDestroyObjModelPlus(&aux->plus[0]);
                }
            }
        }
    }
}


void iftWriteObjModel(  iftObjModel *obj_model, const char *path) {
    if (obj_model == NULL)
        iftError("Object Model is NULL", "iftWriteObjModel");
    if (path == NULL)
        iftError("Pathname is NULL", "iftWriteObjModel");

    char *tmp_dir = iftMakeTempDir("./tmpdir_");

    // writing model
    if (obj_model->model != NULL) {
        char *model_path = iftJoinPathnames(2, tmp_dir, "model.npy");
        iftWriteFImage(obj_model->model, model_path);
        iftFree(model_path);
    }

    // writing label map
    if (obj_model->label_map != NULL) {
        char *label_map_path = NULL;
        if (iftIs3DImage(obj_model->label_map))
            label_map_path = iftJoinPathnames(2, tmp_dir, "label_map.scn");
        else label_map_path = iftJoinPathnames(2, tmp_dir, "label_map.pgm");

        iftWriteImageByExt(obj_model->label_map, label_map_path);
        iftFree(label_map_path);
    }

    // writing reference image
    if (obj_model->ref_img != NULL) {
        char *ref_img_path = NULL;
        if (iftIs3DImage(obj_model->ref_img))
            ref_img_path = iftJoinPathnames(2, tmp_dir, "ref_img.scn");
        else ref_img_path = iftJoinPathnames(2, tmp_dir, "ref_img.pgm");

        iftWriteImageByExt(obj_model->ref_img, ref_img_path);
        iftFree(ref_img_path);
    }


    // writing a Json with information
    char *info_path = iftJoinPathnames(2, tmp_dir, "info.json");
    iftJson *info = iftCreateJsonRoot();
    iftAddIntArrayToJson(info, "labels", obj_model->labels);

    // writing (reference) image depth in bits
    iftAddIntToJson(info, "img-depth", obj_model->img_depth);

    // writing segmentation policy
    if (obj_model->seg_pol == IFT_ONE_BY_ONE) {
        iftAddStringToJson(info, "segmentation-policy", "ONE_BY_ONE");
    }
    else if (obj_model->seg_pol == IFT_ALL_AT_ONCE) {
        iftAddStringToJson(info, "segmentation-policy", "ALL_AT_ONCE");
    }

    // writing object model plus
    if (obj_model->plus != NULL) {
        iftAddIntToJson(info, "n_plus", obj_model->n_plus);
        iftAddJDictReferenceToJson(info, "plus", iftCreateJDict());
        char *plus_dir = iftJoinPathnames(2, tmp_dir, "plus");

        if (obj_model->seg_pol == IFT_ONE_BY_ONE) {
            int n_objs = obj_model->labels->n;

            // a single object model plus for all objects
            if (_iftIsOneByOneWithSingleObjectModelPlus(obj_model)) {
                _iftWriteObjModelPlus(obj_model->plus[0], "single", info, plus_dir);
            }
                // one object model plus for each object
            else {
                for (int o = 0; o < n_objs; o++) {
                    char obj_key[32];
                    sprintf(obj_key, "obj_%d", obj_model->labels->val[o]);
                    _iftWriteObjModelPlus(obj_model->plus[o], obj_key, info, plus_dir);
                }
            }
        }
        else if (obj_model->seg_pol == IFT_ALL_AT_ONCE) {
            _iftWriteObjModelPlus(obj_model->plus[0], "all", info, plus_dir);
        }
        iftFree(plus_dir);
    }

    iftWriteJson(info, info_path);
    iftDestroyJson(&info);
    iftFree(info_path);

    iftZipDirContent(tmp_dir, path);

    iftRemoveDir(tmp_dir);
    iftFree(tmp_dir);
}


iftObjModel *iftReadObjModel(const char *path) {
    if (path == NULL)
        iftError("Pathname is NULL", "iftReadObjModel");

    char *tmp_dir = iftMakeTempDir("./tmpdir_");
    iftUnzipFile(path, tmp_dir);

    iftObjModel *obj_model = iftCreateObjModel();

    //////// READS MODEL ////////
    char *model_path = iftJoinPathnames(2, tmp_dir, "model.npy");
    if (!iftFileExists(model_path)) {
        iftRemoveDir(tmp_dir);
        iftError("Model not found", "iftReadObjModel");
    }

    obj_model->model = iftReadFImage(model_path);
    iftFree(model_path);
    ////////////////////////////

    //////// READS LABEL MAP ////////
    char *label_map_path = iftJoinPathnames(2, tmp_dir, "label_map.scn");
    if (iftFileExists(label_map_path)) {
        obj_model->label_map = iftReadImage(label_map_path);
    }
    else {
        iftFree(label_map_path);
        label_map_path = iftJoinPathnames(2, tmp_dir, "label_map.pgm");
        if (iftFileExists(label_map_path))
            obj_model->label_map = iftReadImagePNG(label_map_path);
        else {
            iftRemoveDir(tmp_dir);
            iftError("Label Map not found", "iftReadObjModel");
        }
    }
    iftFree(label_map_path);
    ////////////////////////////////


    //////// READING REFERENCE IMAGE ////////
    char *ref_img_path = iftJoinPathnames(2, tmp_dir, "ref_img.scn");
    if (iftFileExists(ref_img_path)) {
        obj_model->ref_img = iftReadImage(ref_img_path);
    }
    else {
        iftFree(ref_img_path);
        ref_img_path = iftJoinPathnames(2, tmp_dir, "ref_img.pgm");
        if (iftFileExists(ref_img_path))
            obj_model->ref_img = iftReadImagePNG(ref_img_path);
        else {
            obj_model->ref_img = NULL; // no Reference Image
        }
    }
    iftFree(ref_img_path);
    ////////////////////////////////


    //////// READING JSON INFO ////////
    char *info_path = iftJoinPathnames(2, tmp_dir, "info.json");
    if (!iftFileExists(info_path)) {
        iftRemoveDir(tmp_dir);
        iftError("Info Json file \"info.json\" not found", "iftReadObjModel");
    }

    iftJson *info     = iftReadJson(info_path);
    obj_model->labels = iftGetJIntArray(info, "labels");

    // reading (reference) image depth in bits
    obj_model->img_depth = iftGetJInt(info, "img-depth");


    // reading segmentation policy
    char *seg_pol = iftGetJString(info, "segmentation-policy");
    if (iftCompareStrings(seg_pol, "ONE_BY_ONE")) {
        obj_model->seg_pol = IFT_ONE_BY_ONE;
    }
    else if (iftCompareStrings(seg_pol, "ALL_AT_ONCE")) {
        obj_model->seg_pol = IFT_ALL_AT_ONCE;
    }
    iftFree(seg_pol);


    if (iftJsonContainKey("plus", info)) {
        obj_model->n_plus = iftGetJInt(info, "n_plus"); // reads the number of object model plus

        char *plus_dir = iftJoinPathnames(2, tmp_dir, "plus");

        // READ OBJECT MODEL PLUS 
        if (obj_model->seg_pol == IFT_ONE_BY_ONE) {
            obj_model->plus = (iftObjModelPlus**) iftAlloc(obj_model->n_plus, sizeof(iftObjModelPlus*));
            int n_objs      = obj_model->labels->n; // number of objects from the model

            // a single object model plus for all objects
            if (_iftIsOneByOneWithSingleObjectModelPlus(obj_model)) {
                obj_model->plus[0] = _iftReadObjModelPlus(info, "single", plus_dir);
            }
                // one object model plus for each object
            else {
                for (int o = 0; o < n_objs; o++) {
                    char obj_key[32];
                    sprintf(obj_key, "obj_%d", obj_model->labels->val[o]);

                    obj_model->plus[o] = _iftReadObjModelPlus(info, obj_key, plus_dir);
                }
            }
        }
        else if (obj_model->seg_pol == IFT_ALL_AT_ONCE) {
            obj_model->plus = (iftObjModelPlus**) iftAlloc(1, sizeof(iftObjModelPlus*));
            obj_model->plus[0] = _iftReadObjModelPlus(info, "all", plus_dir);
        }
        iftFree(plus_dir);
    }

    iftDestroyJson(&info);
    iftFree(info_path);
    ////////////////////////////////


    iftRemoveDir(tmp_dir);
    iftFree(tmp_dir);

    return obj_model;
}




iftObjModel *iftBuildSOSM(  iftFileSet *label_set,   iftIntArray *labels,   iftImage *ref_img,
                          iftObjProcPolicy seg_pol) {
    if (label_set == NULL)
        iftError("Label Set is NULL", "iftBuildSOSM");
    if (label_set->n == 0)
        iftError("Label Set is Empty", "iftBuildSOSM");
    if (labels == NULL)
        iftError("Array of Object Labels is NULL", "iftBuildSOSM");
    if (labels->n == 0)
        iftError("Label Array is Empty", "iftBuildSOSM");
    if (ref_img == NULL)
        iftError("Reference Image is NULL", "iftBuildSOSM");


    iftImage *first_img    = iftReadImageByExt(label_set->files[0]->path);
    iftObjModel *obj_model = iftCreateObjModel();
    obj_model->ref_img     = iftCopyImage(ref_img);
    obj_model->seg_pol     = seg_pol;
    obj_model->labels      = iftCreateIntArray(labels->n);
    iftCopyIntArray(obj_model->labels->val, labels->val, labels->n);

    obj_model->model       = iftCreateFImage(first_img->xsize, first_img->ysize, first_img->zsize);
    obj_model->label_map   = iftCreateImage(first_img->xsize, first_img->ysize, first_img->zsize);
    iftCopyVoxelSize(first_img, obj_model->model);
    iftCopyVoxelSize(first_img, obj_model->label_map);
    iftDestroyImage(&first_img);

    int max_range        = iftNormalizationValue(iftMaximumValue(ref_img));
    obj_model->img_depth = iftLog(max_range+1, 2);

    // indexes the object labels into a dict
    iftDict *label_idxs = iftCreateDictWithApproxSize(labels->n);
    iftFImage **models  = (iftFImage**) iftAlloc(labels->n, sizeof(iftFImage*));
    for (int o = 0; o < labels->n; o++) {
        iftInsertIntoDict(labels->val[o], o, label_idxs);
        models[o] = iftCreateFImage(obj_model->model->xsize, obj_model->model->ysize, obj_model->model->zsize);
    }

    // for each label_img
#pragma omp parallel for schedule(auto)
    for (size_t i = 0; i < label_set->n; i++) {
        const char *label_path = label_set->files[i]->path;
        iftImage *label_img    = iftReadImageByExt(label_path);

        /* Only used for dev */
        /*
        if (!iftIsDomainEqual(obj_model->label_map, label_img)) {
            iftError("Images with different domains\n" \
                     "First Image: %dx%dx%d\n" \
                     "Current Image: %s\n" \
                     "%dx%dx%d", "iftBuildSOSM",
                     obj_model->label_map->xsize, obj_model->label_map->ysize, obj_model->label_map->zsize,
                     label_path,
                     label_img->xsize, label_img->ysize, label_img->zsize);
        }
        */

        for (int p = 0; p < label_img->n; p++) {
            // if the label is in the label array
            if (iftDictContainKey(label_img->val[p], label_idxs, NULL)) {
                int o = iftGetLongValFromDict(label_img->val[p], label_idxs);
#pragma omp atomic
                models[o]->val[p]++;
            }
        }

        iftDestroyImage(&label_img);
    }


    ///// builds the prob. atlas /////
    for (int p = 0; p < obj_model->model->n; p++) {
        for (int o = 0; o < labels->n; o++) {
            float prob = models[o]->val[p] / (label_set->n * 1.0);

            if (prob > obj_model->model->val[p]) {
                obj_model->model->val[p]     = prob;
                obj_model->label_map->val[p] = labels->val[o];
            }
        }
    }
    iftDestroyDict(&label_idxs);
    for (int o = 0; o < labels->n; o++)
        iftDestroyFImage(&models[o]);
    iftFree(models);

    return obj_model;
}


iftObjModel *iftBuildSOSM_A(  iftFileSet *label_set,   iftIntArray *labels,   iftImage *ref_img,
                            iftObjProcPolicy seg_pol,   iftFileSet *markers_paths,
                            iftClassificationAlg clf_alg, float smooth_factor) {
    if (markers_paths == NULL)
        iftError("File set of the Markers is NULL", "iftBuildSOSM_A");
    if ((smooth_factor < 0.0) || (smooth_factor > 1.0))
        iftError("Invalid Smooth Factor: %f... Try [0, 1]", "iftBuildSOSM_A", smooth_factor);

    iftObjModel *obj_model = iftBuildSOSM(label_set, labels, ref_img, seg_pol);

    // LEARNING THE TEXTURE (APPEARANCE) CLASSIFIERS
    if (seg_pol == IFT_ONE_BY_ONE) {
        // if an only marker path is passed, a single classifier will be trained and used for all objects
        obj_model->n_plus = markers_paths->n;
        obj_model->plus   = (iftObjModelPlus**) iftAlloc(obj_model->n_plus, sizeof(iftObjModelPlus*));

        // for each marker file
        for (int o = 0; o < markers_paths->n; o++) {
            obj_model->plus[o] = _iftTrainObjModelPlusByOPF(markers_paths->files[o]->path, ref_img, smooth_factor);
        }
    }
    else if (seg_pol == IFT_ALL_AT_ONCE) {
        obj_model->n_plus  = 1;
        obj_model->plus    = (iftObjModelPlus**) iftAlloc(obj_model->n_plus, sizeof(iftObjModelPlus*));
        obj_model->plus[0] = _iftTrainObjModelPlusByOPF(markers_paths->files[0]->path, ref_img, smooth_factor);
    }

    return obj_model;
}


iftLabeledSet *iftFindModelSeeds(  iftFImage *model,   iftImage *model_label_map,
                                 double e_radius, double d_radius, iftSet **out_bg_certainty_region) {
    if (model == NULL)
        iftError("Model is NULL", "iftFindModelSeeds");
    if (model_label_map == NULL)
        iftError("Label Map is NULL", "iftFindModelSeeds");


    // INNER SEEDS (Certain Region Border of the Target Object)
    iftSet *inner_seeds      = NULL;
    iftImage *model_obj_mask = iftFThreshold(model, 1, IFT_INFINITY_FLT, 1);
    iftImage *erosion        = iftErodeBin(model_obj_mask, &inner_seeds, e_radius);
    iftDestroyImage(&model_obj_mask);
    iftDestroyImage(&erosion);

    // OUTER SEEDS (Certain Region Border of the Background)
    iftSet *outer_seeds     = NULL;
    iftImage *model_bg_mask = iftFThreshold(model, 0, 0, 1);
    iftImage *dilation      = iftErodeBin(model_bg_mask, &outer_seeds, d_radius);
    iftDestroyImage(&model_bg_mask);
    model_bg_mask = dilation;

    // gets the certainty region (prob. 1) from the background
    if (out_bg_certainty_region != NULL) {
        for (int p = 0; p < model_label_map->n; p++) {
            if (model_bg_mask->val[p] == 1) {
                iftInsertSet(out_bg_certainty_region, p);
            }
        }
    }
    iftDestroyImage(&model_bg_mask);


    iftLabeledSet *all_seeds = NULL;

    iftSet *S = inner_seeds;
    while (S != NULL) {
        int p     = iftRemoveSet(&S);
        int label = model_label_map->val[p];
        iftInsertLabeledSet(&all_seeds, p, label);
    }
    iftSet *T = outer_seeds;
    while (T != NULL) {
        int p     = iftRemoveSet(&T);
        int label = 0; // background
        iftInsertLabeledSet(&all_seeds, p, label);
    }

    return all_seeds;
}


iftLabeledSet *iftFindFilteredModelSeeds(  iftFImage *model,   iftImage *model_label_map,
                                           iftImage *clf_mask, double e_radius, double d_radius,
                                         iftSet **out_bg_certainty_region) {
    if (model == NULL)
        iftError("Model is NULL", "iftFindFilteredModelSeeds");
    if (model_label_map == NULL)
        iftError("Label Map is NULL", "iftFindFilteredModelSeeds");
    if (clf_mask == NULL)
        iftError("Classification Mask (membership map) is NULL", "iftFindFilteredModelSeeds");


    ///// INNER SEEDS (Intersection Certain Region of the Model with the Classification Mask)
    iftSet *inner_seeds = NULL;

    iftImage *model_obj_mask = iftFThreshold(model, 1, IFT_INFINITY_FLT, 1);
    iftImage *intersec       = iftAnd(model_obj_mask, clf_mask);
    iftImage *sub            = iftSub(model_obj_mask, intersec);
    iftDestroyImage(&model_obj_mask);
    iftImage *erosion = iftErodeBin(intersec, &inner_seeds, e_radius);
    iftDestroyImage(&intersec);
    iftDestroyImage(&erosion);
    //////

    ///// OUTER SEEDS (Pixels that the model confirm that are object, and the Bin Mask confirms that are BG)
    iftSet *outer_seeds     = NULL;
    iftImage *model_bg_mask = iftFThreshold(model, 0, 0, 1);
    iftImage *dilation      = iftErodeBin(model_bg_mask, &outer_seeds, d_radius);
    iftDestroyImage(&model_bg_mask);
    iftDestroyImage(&dilation);

    iftMaskImageToSet(sub, &outer_seeds);

    iftDestroyImage(&sub);
    //////

    // TODO: FORBIDDEN REGION FOR DELINEATION
    iftLabeledSet *all_seeds = NULL;

    iftSet *S = inner_seeds;
    while (S != NULL) {
        int p     = iftRemoveSet(&S);
        int label = model_label_map->val[p];
        iftInsertLabeledSet(&all_seeds, p, label);
    }
    iftSet *T = outer_seeds;
    while (T != NULL) {
        int p     = iftRemoveSet(&T);
        iftInsertLabeledSet(&all_seeds, p, 0);
    }

    return all_seeds;
}


/**
 * @brief Extracts (only) the model of the object with label <target_label>
 * @author Samuel Martins
 * @date Sep 7, 2016
 * 
 * @param  model            (Multi-Object) Model.
 * @param  label_map        Label Map of the model.
 * @param  target_obj_label Target object to be extracted.
 * @return                  The (single) model of the object <target_label>
 */


iftImage *iftSegmentBySOSM(iftImage *test_img,   iftObjModel *obj_model, iftGradientAlg grad_alg,
                           float beta, float e_radius, float d_radius) {
    if (test_img == NULL)
        iftError("Input image to be segmented is NULL", "iftSegmentBySOSM");
    if (obj_model == NULL)
        iftError("SOSM is NULL", "iftSegmentBySOSM");

    ///// Compute Final Gradient Image /////
    iftImage *final_grad_img = NULL;

    if (grad_alg == IFT_NONE_GRAD) {
        final_grad_img = test_img;
    }
    else {
        iftImage *test_grad_img = iftComputeGradient(test_img, grad_alg);

        int img_max_range = (1 << obj_model->img_depth) - 1; // 2^depth -1
        iftImage *model_img      = iftFImageToImage(obj_model->model, img_max_range);
        iftImage *model_grad_img = iftComputeGradient(model_img, grad_alg);
        iftDestroyImage(&model_img);

        final_grad_img = _iftCombineGradImages(test_grad_img, model_grad_img, beta, img_max_range);
    }
    ////////////////////////////////////////


    ///// Segmentation /////
    iftImage *seg_img = NULL;

    if (obj_model->seg_pol == IFT_ALL_AT_ONCE) {
        puts("  - Segmenting ALL AT ONCE");
        seg_img = _iftDelineateBySOSM(test_img, final_grad_img, obj_model->model, obj_model->label_map, e_radius, d_radius);
    }
    else if (obj_model->seg_pol == IFT_ONE_BY_ONE) {
        puts("  - Segmenting ONE BY ONE");
        seg_img = iftCreateImage(obj_model->model->xsize, obj_model->model->ysize, obj_model->model->zsize);
        iftCopyVoxelSize(obj_model->model, seg_img);

        // there is no treatment for race condition
        // however, it is expected that the segmented objects are not overlapped
#pragma omp parallel for schedule(auto)
        for (int o = 0; o < obj_model->labels->n; o++) {
            iftFImage *single_model = _iftExtractSingleObjModel(obj_model->model, obj_model->label_map,
                                                                obj_model->labels->val[o]);

            // char tmp[512];
            // iftImage *aux = iftFImageToImage(single_model, 4095);
            // sprintf(tmp, "%d_model.scn", obj_model->labels->val[o]);
            // iftWriteImageByExt(aux, tmp);
            // iftDestroyImage(&aux);

            iftImage *seg_obj_img = _iftDelineateBySOSM(test_img, final_grad_img, single_model,
                                                        obj_model->label_map, e_radius, d_radius);
            // sprintf(tmp, "%d_seg.scn", obj_model->labels->val[o]);
            // iftWriteImageByExt(seg_obj_img, tmp);

            // merging segmentations
#pragma omp parallel for schedule(auto)
            for (int p = 0; p < seg_obj_img->n; p++) {
                if (seg_obj_img->val[p] != 0) {
                    seg_img->val[p] = seg_obj_img->val[p];
                }
            }

            iftDestroyFImage(&single_model);
            iftDestroyImage(&seg_obj_img);
        }
    }
    ///////////////////////


    // DESTROYERS
    if (test_img != final_grad_img)
        iftDestroyImage(&final_grad_img);

    return seg_img;
}


iftImage *iftSegmentBySOSM_A(iftImage *test_img,   iftObjModel *obj_model, iftGradientAlg grad_alg,
                             float alpha, float beta, float e_radius, float d_radius) {
    if (test_img == NULL)
        iftError("Input image to be segmented is NULL", "iftSegmentBySOSM_A");
    if (obj_model == NULL)
        iftError("SOSM is NULL", "iftSegmentBySOSM_A");
    if (obj_model->plus == NULL)
        iftError("Object Model Plus struct is NULL. Use iftSegmentBySOSM instead", "iftSegmentBySOSM_A");

    int img_max_range = (1 << obj_model->img_depth) - 1; // 2^depth -1

    ///// Segmentation /////
    iftImage *seg_img = NULL;

    if (obj_model->seg_pol == IFT_ALL_AT_ONCE) {
        puts("  - Segmenting ALL AT ONCE");
        iftObjModelPlus *plus = obj_model->plus[0];

        seg_img = _iftRunObjModelPlus(test_img, obj_model->model, obj_model->label_map, plus, grad_alg,
                                      alpha, beta, e_radius, d_radius, img_max_range);
    }
    else if (obj_model->seg_pol == IFT_ONE_BY_ONE) {
        puts("  - Segmenting ONE BY ONE");
        seg_img = iftCreateImage(obj_model->model->xsize, obj_model->model->ysize, obj_model->model->zsize);

        // there is no treatment for race condition
        // however, it is expected that the segmented objects are not overlapped
#pragma omp parallel for schedule(auto)
        for (int o = 0; o < obj_model->labels->n; o++) {
            iftObjModelPlus *plus = NULL;
            if (_iftIsOneByOneWithSingleObjectModelPlus(obj_model)) { // one classifier for all objects
                plus = obj_model->plus[0];
                printf("[%d] = %d - 1\n", o, obj_model->labels->val[o]);
            }
            else {
                plus = obj_model->plus[o];
            }

            printf("[%d] = %d - 2\n", o, obj_model->labels->val[o]);
            iftFImage *single_model = _iftExtractSingleObjModel(obj_model->model, obj_model->label_map,
                                                                obj_model->labels->val[o]);
            printf("[%d] = %d - 3\n", o, obj_model->labels->val[o]);

            iftImage *seg_obj_img = _iftRunObjModelPlus(test_img, single_model, obj_model->label_map, plus,
                                                        grad_alg, alpha, beta, e_radius, d_radius, img_max_range);
            printf("[%d] = %d - 4\n", o, obj_model->labels->val[o]);
            // char tmp[512];
            // sprintf(tmp, "%d_seg.scn", obj_model->labels->val[o]);
            // iftWriteImageByExt(seg_obj_img, tmp);

            // merging segmentations
#pragma omp parallel for schedule(auto)
            for (int p = 0; p < seg_obj_img->n; p++) {
                if (seg_obj_img->val[p] != 0) {
                    seg_img->val[p] = seg_obj_img->val[p];
                }
            }
            printf("[%d] = %d - 5\n", o, obj_model->labels->val[o]);

            iftDestroyFImage(&single_model);
            iftDestroyImage(&seg_obj_img);
        }
    }
    else iftError("Invalid Segmentation policy", "iftSegmentBySOSM_A");

    ///////////////////////


    return seg_img;
}


void iftRegisterObjModelByElastix(iftObjModel *obj_model,   iftImage *fixed_img,
                                  const char *affine_params_path, const char *bspline_params_path) {
    if (obj_model == NULL)
        iftError("Object Model is NULL", "iftRegisterObjModelByElastix");
    if (obj_model->ref_img == NULL)
        iftError("Reference Image from the Object Model is NULL", "iftRegisterObjModelByElastix");
    if (fixed_img == NULL)
        iftError("Fixed Image is NULL", "iftRegisterObjModelByElastix");
    if (affine_params_path == NULL)
        iftError("Affine Parameters path is NULL", "iftRegisterObjModelByElastix");
    if (bspline_params_path == NULL)
        iftError("BSpline Parameters path is NULL", "iftRegisterObjModelByElastix");


    char *tmp_dir     = iftMakeTempDir("./tmpdir_");
    char *base        = iftJoinPathnames(2, tmp_dir, "base");
    iftImage *reg_img = NULL;

    iftRegistrationParams *reg_params = iftSetupElastixRegistation(affine_params_path, bspline_params_path,
                                                                   obj_model->img_depth, base);

    // Registering the Reference Image to the Fixed Image
    reg_img = iftRegisterImage(obj_model->ref_img, fixed_img, reg_params, NULL);
    iftDestroyImage(&reg_img);
    iftDestroyImage(&obj_model->ref_img);

    // Mapping the Model
    char *affine_def_fields_path  = iftConcatStrings(2, base, "_DefFields_Affine.txt");
    char *bspline_def_fields_path = iftConcatStrings(2, base, "_DefFields_BSpline.txt");

    int img_max_range   = (1 << obj_model->img_depth) - 1; // 2^depth -1
    iftImage *model_img = iftFImageToImage(obj_model->model, img_max_range);
    iftDestroyFImage(&obj_model->model);

    reg_img = iftTransformImage(model_img, bspline_def_fields_path, IFT_TRANSFORMIX);
    iftDestroyImage(&model_img);

    obj_model->model = iftImageToFImageMaxVal(reg_img, 1.0);
    iftDestroyImage(&reg_img);

    // Mapping the Model
    iftImage *label_img  = obj_model->label_map;
    obj_model->label_map = iftTransformImage(label_img, bspline_def_fields_path, IFT_TRANSFORMIX);
    iftDestroyImage(&label_img);

    iftFree(affine_def_fields_path);
    iftFree(bspline_def_fields_path);

    iftRemoveDir(tmp_dir);
    iftFree(tmp_dir);
    iftFree(base);

    iftDestroyRegistrationParams(&reg_params);
}
/******************************************************************/










































//=======================================================================================================
/********************** PUBLIC FUNCTIONS *************************/
iftOMDelinAlgData *iftCreateOMDelinAlgData(iftImage *orig, void *custom_data) {
    iftOMDelinAlgData *data = NULL;

    data = (iftOMDelinAlgData*) iftAlloc(1, sizeof(iftOMDelinAlgData));

    data->orig        = orig;
    data->custom_data = custom_data;

    return data;
}


iftOMDelinAlgData *iftCreateOMDelinAlgWatershedData(iftImage *orig, iftImageForest *fst) {
    iftOMDelinAlgData *data;
    iftOMDelinAlgWatershedCustomData *custom_data = NULL;

    custom_data = (iftOMDelinAlgWatershedCustomData*) iftAlloc(1, sizeof(iftOMDelinAlgWatershedCustomData));

    custom_data->border_processed = iftCreateBMap(fst->img->n);
    custom_data->mask = NULL;
    custom_data->fst = fst;

    data = iftCreateOMDelinAlgData(orig, custom_data);

    return data;
}


void iftDestroyOMDelinAlgWatershedData(iftOMDelinAlgData **data) {
    if ((data != NULL) && (*data != NULL)) {
        iftOMDelinAlgWatershedCustomData *custom_data = (*data)->custom_data;

        if (custom_data != NULL)
            iftDestroyBMap(&custom_data->border_processed);
        if (custom_data->mask != NULL)
            iftDestroyBMap(&custom_data->mask);

        iftFree((*data)->custom_data);
        iftFree(*data);
        *data = NULL;
    }
}


iftOMRecogAlgData *iftCreateOMRecogAlgData(void *custom_data, iftOptStrategy opt_str) {
    iftOMRecogAlgData *data = (iftOMRecogAlgData*) iftAlloc(1, sizeof(iftOMRecogAlgData));

    data->opt_str     = opt_str;
    data->custom_data = custom_data;

    if (opt_str == IFT_MAXIMIZATION) {
        data->worst_score = IFT_INFINITY_FLT_NEG;
        data->best_score  = IFT_INFINITY_FLT;
    }
    else if (opt_str == IFT_MINIMIZATION) {
        data->worst_score = IFT_INFINITY_FLT;
        data->best_score  = IFT_INFINITY_FLT_NEG;
    }
    else iftError("Invalid Optimization Strategy", "iftCreateOMRecogAlgData");

    return data;
}


iftOMRecogAlgData *iftCreateOMRecogAlgDataMeanCut(iftImageForest *fst) {
    return iftCreateOMRecogAlgData(NULL, IFT_MAXIMIZATION);
}


void iftDestroyOMRecogDataMeanCut(iftOMRecogAlgData **data) {
    if ((data != NULL) && (*data != NULL)) {
        iftFree(*data);
        *data = NULL;
    }
}


iftObjectModelProb *iftCreateObjectModelProb(void *model, iftAdjRel **ALabels, iftAdjRel *AUncertain,
                                             iftIntArray *labels, iftOMDelinAlgData *delineation_data,
                                             iftOMRecogAlgData *recognition_data, iftBMap *search_region) {
    iftObjectModelProb *prob = NULL;

    prob = (iftObjectModelProb *)iftAlloc(1, sizeof(iftObjectModelProb));

    prob->model            = model;
    prob->seed_adj         = ALabels;
    prob->uncertainty_adj  = AUncertain;
    prob->labels           = labels;
    prob->delineation_data = delineation_data;
    prob->recognition_data = recognition_data;
    prob->search_region    = search_region;

    // Setting Watershed + Mean Cut as the default delineation and recognition algorithms.
    // Change these values outside this function for customized algorithms.
    prob->iftDelinAlgorithm         = iftOMDelinWatershed;
    prob->iftDelinAlgPostProcessing = NULL;
    prob->iftDelinAlgInit           = iftOMDelinWatershedInit;
    prob->iftDelinAlgIterationSetup = iftOMDelinWatershedIterationSetupSingleCenter;
    prob->iftDelinAlgReset          = iftOMDelinWatershedReset;
    prob->iftRecogAlgIterationSetup = NULL;
    prob->iftRecogAlgorithm         = iftOMRecogAlgWatershedMeanCut;

    return prob;
}


void iftDestroyObjectModelProb(iftObjectModelProb **prob){
    if ((prob != NULL) && (*prob != NULL)) {
        iftFree(*prob);
        *prob = NULL;
    }
}
/******************************************************************/



















/**
@file
@brief A description is missing here
*/
iftImage *iftCropObjectLoose(iftImage *bin, int xLSize, int yLSize, int zLSize)
{
    int p, q;
    iftVoxel u, v, c, cResult;
    iftImage *result, *crop;

    iftBoundingBox mbb = iftMinObjectBoundingBox(bin, 1, NULL);
    crop               = iftExtractROI(bin, mbb);
    c = iftGeometricCenterVoxel(crop);
    result = iftCreateImage(xLSize,yLSize,zLSize);
    cResult.x = iftRound((result->xsize - 1) / 2);
    cResult.y = iftRound((result->ysize - 1) / 2);
    cResult.z = iftRound((result->zsize - 1) / 2);

    for (u.z=0, v.z= cResult.z - c.z; u.z < crop->zsize; u.z++,v.z++)
        for (u.y=0, v.y= cResult.y - c.y; u.y < crop->ysize; u.y++,v.y++)
            for (u.x=0, v.x= cResult.x - c.x; u.x < crop->xsize; u.x++,v.x++)
                if (iftValidVoxel(result,v))
                {
                    p = iftGetVoxelIndex(crop,u);
                    q = iftGetVoxelIndex(result,v);
                    result->val[q]= crop->val[p];
                }

    iftCopyVoxelSize(bin, result);
    iftDestroyImage(&crop);

    return(result);
}


/**
 * @brief Finds the Largest Minimum Bounding Box for a given Object from a Image Set.
 * @author Samuel Martins
 * @date Apr 4, 2016
 * @ingroup Image
 *
 * @param img_set
 */
//Takes a directory including 3D images which represent object segmentations contained in
//minimum boundig boxes of sizes (xi, yi, zi) and returns (max(xi), max(yi), max(zi))
iftRefSpace *iftModelDomain(  iftFileSet *label_paths, int obj_label,
                            int *model_xsize, int *model_ysize, int *model_zsize) {
    if (label_paths == NULL)
        iftError("Label Set is NULL", "iftModelDomain");
    if (label_paths->n == 0)
        iftError("Label Set doesn't have images", "iftModelDomain");
    int padding = 1;

    iftRefSpace *ref_info = (iftRefSpace*) iftAlloc(1, sizeof(iftRefSpace));

    iftImage *img = iftReadImage(label_paths->files[0]->path);
    ref_info->xsize = img->xsize;
    ref_info->ysize = img->ysize;
    ref_info->zsize = img->zsize;
    ref_info->dx = img->dx;
    ref_info->dy = img->dy;
    ref_info->dz = img->dz;

    iftBoundingBox max_bb = iftMinObjectBoundingBox(img, obj_label, NULL);

    iftDestroyImage(&img);

    for(int i = 1; i < label_paths->n; i++) {
        img = iftReadImage(label_paths->files[i]->path);

        // TODO: Also check if the image is isotropic and if has the same voxel size
        if ((img->xsize != ref_info->xsize) || (img->ysize != ref_info->ysize) || (img->zsize != ref_info->zsize))
            iftError("Images with different dimensions", "iftModelDomain");
        if (!iftIsVoxelSizeEqual(img, ref_info))
            iftError("Images with different Voxel Sizes", "iftModelDomain");

        iftBoundingBox mbb = iftMinObjectBoundingBox(img, obj_label, NULL);
        max_bb.begin.x = iftMin(max_bb.begin.x, mbb.begin.x);
        max_bb.begin.y = iftMin(max_bb.begin.y, mbb.begin.y);
        max_bb.begin.z = iftMin(max_bb.begin.z, mbb.begin.z);
        max_bb.end.x = iftMax(max_bb.end.x, mbb.end.x);
        max_bb.end.y = iftMax(max_bb.end.y, mbb.end.y);
        max_bb.end.z = iftMax(max_bb.end.z, mbb.end.z);

        iftDestroyImage(&img);
    }
    // padding in each side
    *model_xsize = (max_bb.end.x - max_bb.begin.x + 1) + 2*padding;
    *model_ysize = (max_bb.end.y - max_bb.begin.y + 1) + 2*padding;
    *model_zsize = (max_bb.end.z - max_bb.begin.z + 1) + 2*padding;
    // central point of the model on the Ref. Space
    ref_info->model_center.x = (max_bb.begin.x + max_bb.end.x) / 2;
    ref_info->model_center.y = (max_bb.begin.y + max_bb.end.y) / 2;
    ref_info->model_center.z = (max_bb.begin.z + max_bb.end.z) / 2;

    return ref_info;
}



int iftHasUncertain6Adjacent(iftFImage *model, int i){

    iftVoxel u, v;
    int result = 0, q;
    iftAdjRel  *A = iftSpheric(1.0);

    u.x = iftGetXCoord(model, i);
    u.y = iftGetYCoord(model, i);
    u.z = iftGetZCoord(model, i);

    for(int j = 0; j < A->n; j++)
    {
        v = iftGetAdjacentVoxel(A, u, j);
        q = iftGetVoxelIndex(model, v);
        if (iftFValidVoxel(model, v) && (model->val[q] > 0) && (model->val[q] < 1))
        {
            result = 1;
            break;
        }
    }

    iftDestroyAdjRel(&A);

    return result;
}

// void iftModelAdjacencies(iftFImage *model, iftAdjRel **AIn, iftAdjRel **AOut, iftAdjRel **AUncertain){
//    int nAIn = 0, nAOut = 0, nAUncertain = 0, iIn = 0, iOut = 0, iUncertain = 0, p;
//    iftVoxel center, u;

//    center.x = (model->xsize - 1) / 2;
//    center.y = (model->ysize - 1) / 2;
//    center.z = (model->zsize - 1) / 2;

//    for (int i = 0; i < model->n; i++)
//    {
//       if ((model->val[i] > 0) && (model->val[i] < 1))
//          nAUncertain++;
//       else
//       {
//          if ((model->val[i] == 0) && iftHasUncertain6Adjacent(model, i))
//             nAOut++;          
//          if ((model->val[i] == 1) && iftHasUncertain6Adjacent(model, i))
//             nAIn++;          
//       }
//    }

//    *AIn = iftCreateAdjRel(nAIn);
//    *AOut = iftCreateAdjRel(nAOut);
//    *AUncertain = iftCreateAdjRel(nAUncertain);

//    for (u.z = 0; u.z < model->zsize; u.z++) 
//       for (u.y = 0; u.y < model->ysize; u.y++) 
//          for (u.x = 0; u.x < model->xsize; u.x++) 
//          {
//             p = iftGetVoxelIndex(model, u);            
//             if ((model->val[p] > 0) && (model->val[p] < 1))
//             {
//                (*AUncertain)->dx[iUncertain] = u.x - center.x;
//                (*AUncertain)->dy[iUncertain] = u.y - center.y;
//                (*AUncertain)->dz[iUncertain] = u.z - center.z;
//                iUncertain++;
//             }
//             else
//             {
//                if ((model->val[p] == 0) && iftHasUncertain6Adjacent(model, p))
//                {
//                   (*AOut)->dx[iOut] = u.x - center.x;
//                   (*AOut)->dy[iOut] = u.y - center.y;
//                   (*AOut)->dz[iOut] = u.z - center.z;
//                   iOut++;
//                }
//                if ((model->val[p] == 1) && iftHasUncertain6Adjacent(model, p))
//                {
//                   (*AIn)->dx[iIn] = u.x - center.x;
//                   (*AIn)->dy[iIn] = u.y - center.y;
//                   (*AIn)->dz[iIn] = u.z - center.z;
//                   iIn++;
//                }
//             }
//          }
// }


void iftModelAdjacencies(iftFImage *model, iftAdjRel **AIn, iftAdjRel **AOut, iftAdjRel **AUncertain,
                         double e_radius, double d_radius){
    int nAIn = 0, nAOut = 0, nAUncertain = 0, iIn = 0, iOut = 0, iUncertain = 0, p;
    iftVoxel center, u;

    center.x = model->xsize / 2;
    center.y = model->ysize / 2;
    center.z = model->zsize / 2;

    iftSet *S        = NULL;
    iftImage *mask   = NULL;
    iftImage *c_mask = NULL;
    iftImage *erode  = NULL;
    iftImage *dilate = NULL;
    iftImage *uncentainty = NULL;

    // IN
    mask  = iftFThreshold(model, 1, IFT_INFINITY_FLT, 1);
    erode = iftErodeBin(mask, &S, e_radius);

    nAIn = iftSetSize(S);
    if (*AIn != NULL)
        iftDestroyAdjRel(AIn);

    *AIn = iftCreateAdjRel(nAIn);
    iftDestroyImage(&mask);

    while (S != NULL) {
        p = iftRemoveSet(&S);
        u = iftFGetVoxelCoord(model, p);
        (*AIn)->dx[iIn] = u.x - center.x;
        (*AIn)->dy[iIn] = u.y - center.y;
        (*AIn)->dz[iIn] = u.z - center.z;
        iIn++;
    }

    // OUT
    mask  = iftFThreshold(model, 0, 0, 1);
    c_mask = iftComplement(mask);
    dilate = iftDilateBin(c_mask, &S, d_radius);

    nAOut = iftSetSize(S);
    if (*AOut != NULL)
        iftDestroyAdjRel(AOut);

    *AOut = iftCreateAdjRel(nAOut);
    iftDestroyImage(&c_mask);
    iftDestroyImage(&mask);

    while (S != NULL) {
        p = iftRemoveSet(&S);
        u = iftFGetVoxelCoord(model, p);
        (*AOut)->dx[iOut] = u.x - center.x;
        (*AOut)->dy[iOut] = u.y - center.y;
        (*AOut)->dz[iOut] = u.z - center.z;
        iOut++;
    }

    // UNCERTAINTY
    uncentainty = iftSub(dilate, erode);
    iftDestroyImage(&erode);
    iftDestroyImage(&dilate);

    for (int p = 0; p < uncentainty->n; p++) {
        if (uncentainty->val[p] != 0)
            nAUncertain++;
    }

    if (*AUncertain != NULL)
        iftDestroyAdjRel(AUncertain);
    *AUncertain = iftCreateAdjRel(nAUncertain);

    for (int p = 0; p < uncentainty->n; p++) {
        if (uncentainty->val[p] != 0) {
            u = iftFGetVoxelCoord(model, p);
            (*AUncertain)->dx[iUncertain] = u.x - center.x;
            (*AUncertain)->dy[iUncertain] = u.y - center.y;
            (*AUncertain)->dz[iUncertain] = u.z - center.z;
            iUncertain++;
        }
    }

    iftDestroyImage(&uncentainty);
}


//Computes the fuzzy model, by averaging the sum of pixels of every segmented image
// void iftFuzzyByAveraging(iftFImage *model, fileList *imageFiles){

//    iftImage *image, *croppedImage;
//    iftVoxel centerCrop, centerModel, translation, cropVoxel, modelVoxel;
//    int pCrop, pModel;

//    centerModel.x = (model->xsize - 1) / 2;
//    centerModel.y = (model->ysize - 1) / 2;
//    centerModel.z = (model->zsize - 1) / 2;

//    for(int i = 0; i < imageFiles->n; i++) {
//       image = iftReadImage(imageFiles->filesRoutes[i]);
//       iftBoundingBox mbb = iftMinObjectBoundingBox(image, 1, NULL);
//       croppedImage       = iftExtractROI(image, mbb);

//       centerCrop = iftGeometricCenterVoxel(croppedImage);         
//       translation.x = centerModel.x - centerCrop.x;
//       translation.y = centerModel.y - centerCrop.y;
//       translation.z = centerModel.z - centerCrop.z;

//       for (cropVoxel.z=0; cropVoxel.z < croppedImage->zsize; cropVoxel.z++) 
//          for (cropVoxel.y=0; cropVoxel.y < croppedImage->ysize; cropVoxel.y++) 
//             for (cropVoxel.x=0; cropVoxel.x < croppedImage->xsize; cropVoxel.x++) {
//            pCrop = iftGetVoxelIndex(croppedImage, cropVoxel);
//                modelVoxel.x = cropVoxel.x + translation.x;
//                modelVoxel.y = cropVoxel.y + translation.y;
//                modelVoxel.z = cropVoxel.z + translation.z;
//                pModel = iftGetVoxelIndex(model, modelVoxel);
//                if (iftFValidVoxel(model, modelVoxel))
//                   model->val[pModel] = model->val[pModel] + croppedImage->val[pCrop];                  
//             }
//       iftDestroyImage(&image);
//       iftDestroyImage(&croppedImage); 
//    }

//    for (int k=0; k < model->n; k++)
//       model->val[k] = model->val[k] / imageFiles->n;

// }

//Computes the fuzzy model, by computing the distance transform of
//every image and adding the results

// void iftFuzzyByDistanceTransform(iftFImage *model, fileList *imageFiles){

//    iftImage *image, *croppedImage;
//    iftFImage *distanceImage;
//    iftAdjRel *adjacency;
//    float sigmoidAlfa = 1.0;
//    float value, thresholdIn = 0.85, thresholdOut = 0.1;

//    adjacency = iftSpheric(sqrtf(3.0));
//    for(int i = 0; i < imageFiles->n; i++) 
//    {
//       image = iftReadImage(imageFiles->filesRoutes[i]);
//       croppedImage = iftCropObjectLoose(image, model->xsize, model->ysize, model->zsize); 
//       distanceImage = iftSignedDistTrans(croppedImage, adjacency);  

//       for (int j = 0; j < distanceImage->n; j++)
//          model->val[j] = model->val[j] + distanceImage->val[j];                  

//       printf("%s\n", imageFiles->filesNames[i]);
//       iftDestroyImage(&image);
//       iftDestroyImage(&croppedImage);
//       iftDestroyFImage(&distanceImage); 
//    }
//    iftDestroyAdjRel(&adjacency); 

//    for (int k=0; k < model->n; k++)
//    {
//       value = iftSigmoidalValue((model->val[k]/imageFiles->n), sigmoidAlfa);
//       if (value > thresholdIn) value = 1;
//       if (value < thresholdOut) value = 0;
//       model->val[k] = value;
//    }

// }

//Takes a directory including 3D images which represent object segmentations,
//aligns them by their geometric center and computes a fuzzy model
// void iftComputeFuzzyModel(char *dirname, iftFImage *model, char option){

//    fileList *imageFiles;

//    imageFiles = iftGetFiles(dirname, ".scn");
//    switch (option){
//       case 'A': 
//          iftFuzzyByAveraging(model, imageFiles);
//          break;
//       case 'D': 
//          iftFuzzyByDistanceTransform(model, imageFiles);
//          break;
//       case 'G': 
//          iftFuzzyByAveragingNoGCCentering(model, imageFiles);
//          break;
//    }
//    iftDestroyFileList(&imageFiles);

// }

void iftDestroyFuzzyModelExtended(iftFuzzyModelExtended **fuzzyModel){
    if (fuzzyModel != NULL){
        if ((*fuzzyModel)->model != NULL)
            iftDestroyFImage(&((*fuzzyModel)->model));
        if ((*fuzzyModel)->AIn != NULL)
            iftDestroyAdjRel(&((*fuzzyModel)->AIn));
        if ((*fuzzyModel)->AOut != NULL)
            iftDestroyAdjRel(&((*fuzzyModel)->AOut));
        if ((*fuzzyModel)->AUncertain != NULL)
            iftDestroyAdjRel(&((*fuzzyModel)->AUncertain));
        // if ((*fuzzyModel)->ref_info != NULL)
        //    iftFree((*fuzzyModel)->ref_info);
        *fuzzyModel = NULL;
    }
}

void iftWriteFuzzyModelExtended(  iftFuzzyModelExtended *fuzzyModel, const char *filename){
    FILE       *fp=NULL;

    fp = fopen(filename,"wb");
    if (fp == NULL){
        iftError(MSG_FILE_OPEN_ERROR, "iftWriteFuzzyModelExtended");
    }

    // target label
    fprintf(fp,"%d\n",fuzzyModel->obj_label);

    fprintf(fp,"%d\n",fuzzyModel->AIn->n);
    for (int i=0; i < fuzzyModel->AIn->n; i++)
        fprintf(fp,"%d %d %d\n",fuzzyModel->AIn->dx[i],fuzzyModel->AIn->dy[i],fuzzyModel->AIn->dz[i]);

    fprintf(fp,"%d\n",fuzzyModel->AOut->n);
    for (int i=0; i < fuzzyModel->AOut->n; i++)
        fprintf(fp,"%d %d %d\n",fuzzyModel->AOut->dx[i],fuzzyModel->AOut->dy[i],fuzzyModel->AOut->dz[i]);

    fprintf(fp,"%d\n",fuzzyModel->AUncertain->n);
    for (int i=0; i < fuzzyModel->AUncertain->n; i++)
        fprintf(fp,"%d %d %d\n",fuzzyModel->AUncertain->dx[i],fuzzyModel->AUncertain->dy[i],fuzzyModel->AUncertain->dz[i]);

    fprintf(fp,"%d %d %d\n",fuzzyModel->start.x,fuzzyModel->start.y,fuzzyModel->start.z);
    fprintf(fp,"%d %d %d\n",fuzzyModel->end.x,fuzzyModel->end.y,fuzzyModel->end.z);

    fprintf(fp,"%5.4f %5.4f %d\n",fuzzyModel->mean,fuzzyModel->deviation,fuzzyModel->mode);

    // save reference space info
    fprintf(fp, "%d %d %d\n", fuzzyModel->ref_info->xsize, fuzzyModel->ref_info->ysize, fuzzyModel->ref_info->zsize);
    fprintf(fp, "%f %f %f\n", fuzzyModel->ref_info->dx, fuzzyModel->ref_info->dy, fuzzyModel->ref_info->dz);
    fprintf(fp, "%d %d %d\n", fuzzyModel->ref_info->model_center.x, fuzzyModel->ref_info->model_center.y, fuzzyModel->ref_info->model_center.z);

    fprintf(fp,"FSCN\n");
    fprintf(fp,"%d %d %d\n",fuzzyModel->model->xsize,fuzzyModel->model->ysize,fuzzyModel->model->zsize);
    fprintf(fp,"%f %f %f\n",fuzzyModel->model->dx,fuzzyModel->model->dy,fuzzyModel->model->dz);
    fwrite(fuzzyModel->model->val,sizeof(float),fuzzyModel->model->n,fp);

    fclose(fp);
}

iftFuzzyModelExtended *iftReadFuzzyModelExtended(const char *filename){
    iftFuzzyModelExtended *fuzzyModel;
    iftFImage  *img=NULL;
    FILE       *fp=NULL;
    char        type[10];
    int         xsize,ysize,zsize,n;
    long        pos;

    fp = fopen(filename,"rb");
    if (fp == NULL){
        iftError(MSG_FILE_OPEN_ERROR, "iftReadFuzzyModelExtended");
    }

    fuzzyModel = (iftFuzzyModelExtended *) iftAlloc(1,sizeof(iftFuzzyModelExtended));

    if (fscanf(fp,"%d",&fuzzyModel->obj_label)!=1)
        iftError("Error when reading obj label","iftReadFuzzyModel AIn");

    if (fscanf(fp,"%d",&n)!=1)
        iftError("Reading error","iftReadFuzzyModel AIn");
    fuzzyModel->AIn = iftCreateAdjRel(n);

    for (int i=0; i < fuzzyModel->AIn->n; i++)
        if (fscanf(fp,"%d %d %d",&(fuzzyModel->AIn->dx[i]),&(fuzzyModel->AIn->dy[i]),&(fuzzyModel->AIn->dz[i]))!=3)
            iftError("Reading error","iftReadFuzzyModel AIn");

    if (fscanf(fp,"%d",&n)!=1)
        iftError("Reading error","iftReadFuzzyModel AOut");
    fuzzyModel->AOut = iftCreateAdjRel(n);

    for (int i=0; i < fuzzyModel->AOut->n; i++)
        if (fscanf(fp,"%d %d %d",&(fuzzyModel->AOut->dx[i]),&(fuzzyModel->AOut->dy[i]),&(fuzzyModel->AOut->dz[i]))!=3)
            iftError("Reading error","iftReadFuzzyModel AOut");

    if (fscanf(fp,"%d",&n)!=1)
        iftError("Reading error","iftReadFuzzyModel AUncertain");
    fuzzyModel->AUncertain = iftCreateAdjRel(n);

    for (int i=0; i < fuzzyModel->AUncertain->n; i++)
        if (fscanf(fp,"%d %d %d",&(fuzzyModel->AUncertain->dx[i]),&(fuzzyModel->AUncertain->dy[i]),&(fuzzyModel->AUncertain->dz[i]))!=3)
            iftError("Reading error","iftReadFuzzyModel AUncertain");

    if (fscanf(fp,"%d %d %d",&(fuzzyModel->start.x),&(fuzzyModel->start.y),&(fuzzyModel->start.z))!=3)
        iftError("Reading error","iftReadFuzzyModel Start");

    if (fscanf(fp,"%d %d %d",&(fuzzyModel->end.x),&(fuzzyModel->end.y),&(fuzzyModel->end.z))!=3)
        iftError("Reading error","iftReadFuzzyModel End");

    if (fscanf(fp,"%f %f %d",&(fuzzyModel->mean),&(fuzzyModel->deviation),&(fuzzyModel->mode))!=3)
        iftError("Reading error","iftReadStatisticModelSimple Parameters");

    // Reference Space Info
    fuzzyModel->ref_info = (iftRefSpace*) iftAlloc(1, sizeof(iftRefSpace));
    if (fscanf(fp, "%d %d %d", &fuzzyModel->ref_info->xsize, &fuzzyModel->ref_info->ysize, &fuzzyModel->ref_info->zsize) != 3)
        iftError("Reading Reference Space Dimensions", "iftReadStatisticModelSimple Parameters");
    if (fscanf(fp, "%f %f %f", &fuzzyModel->ref_info->dx, &fuzzyModel->ref_info->dy, &fuzzyModel->ref_info->dz) != 3)
        iftError("Reading Voxel Size of the Reference Space", "iftReadStatisticModelSimple Parameters");
    if (fscanf(fp, "%d %d %d", &fuzzyModel->ref_info->model_center.x, &fuzzyModel->ref_info->model_center.y, &fuzzyModel->ref_info->model_center.z) != 3)
        iftError("Reading Center Position of the Model on the Ref. Space", "iftReadStatisticModelSimple Parameters");
    printf("ref_info sizes: (%d, %d, %d)\n", fuzzyModel->ref_info->xsize, fuzzyModel->ref_info->ysize, fuzzyModel->ref_info->zsize);
    printf("ref_info disp: (%lf, %lf, %lf)\n", fuzzyModel->ref_info->dx, fuzzyModel->ref_info->dy, fuzzyModel->ref_info->dz);
    printf("ref_info model center: (%d, %d, %d)\n", fuzzyModel->ref_info->model_center.x, fuzzyModel->ref_info->model_center.y, fuzzyModel->ref_info->model_center.z);

    if (fscanf(fp,"%s",type)!=1) iftError("Reading error","iftReadFuzzyModel Model");
    if((strcmp(type,"FSCN")==0)){
        if (fscanf(fp,"%d %d %d",&xsize,&ysize,&zsize)!=3) iftError("Reading error","iftReadFuzzyModel Model");
        img = iftCreateFImage(xsize,ysize,zsize);
        if (fscanf(fp,"%f %f %f",&img->dx,&img->dy,&img->dz)!=3) iftError("Reading error","iftReadFuzzyModel Model");
        pos = ftell(fp);
        fseek(fp,(pos+1)*sizeof(char),SEEK_SET);
        if(fread(img->val,sizeof(float),img->n,fp)!=img->n) iftError("Reading error","iftReadFuzzyModel Model");
        fuzzyModel->model = img;
    }else{
        iftError("Invalid model type: %s","iftReadFuzzyModel Model", type);
    }

    fclose(fp);
    return(fuzzyModel);

}

void iftModelPositionsFuzzyExtended(iftFileSet *img_paths, iftFileSet *label_paths, iftFuzzyModelExtended *fuzzyModel){
    iftImage *originalImage, *croppedImage, *instance;
    iftVoxel cInstance, cCrop, pos, dMin, dMax, dAct;

    dMin.x = dMin.y = dMin.z = IFT_INFINITY_INT;
    dMax.x = dMax.y = dMax.z = IFT_INFINITY_INT_NEG;

    for(int i = 0; i < img_paths->n; i++)
    {
        originalImage = iftReadImage(img_paths->files[i]->path);
        croppedImage = iftExtractGreyObjectPos(originalImage, &pos);
        iftDestroyImage(&originalImage);
        cCrop = iftGeometricCenterVoxel(croppedImage);
        iftDestroyImage(&croppedImage);

        char *img_key = iftBasename(img_paths->files[i]->path);
        for(int j = 0; j < label_paths->n; j++) {
            char *label_key = iftBasename(label_paths->files[j]->path);
            // puts(img_paths->files[i]->path);
            // puts(label_paths->files[j]->path);
            // puts("");
            if (iftCompareStrings(img_key, label_key)) {
                printf("%s ---> %s\n", img_key, label_key);
                instance = iftReadImage(label_paths->files[j]->path);
                break;
            }
            iftFree(label_key);
        }
        iftFree(img_key);
        cInstance = iftGeometricCenterVoxel(instance);

        dAct.x = cInstance.x - cCrop.x - pos.x;
        dAct.y = cInstance.y - cCrop.y - pos.y;
        dAct.z = cInstance.z - cCrop.z - pos.z;
        dMin.x = iftMin(dAct.x, dMin.x);
        dMin.y = iftMin(dAct.y, dMin.y);
        dMin.z = iftMin(dAct.z, dMin.z);
        dMax.x = iftMax(dAct.x, dMax.x);
        dMax.y = iftMax(dAct.y, dMax.y);
        dMax.z = iftMax(dAct.z, dMax.z);

        iftDestroyImage(&instance);
    }

    fuzzyModel->start = dMin;
    fuzzyModel->end = dMax;
}


void iftModelPositionsProbAtlas(iftFileSet *img_paths, iftFileSet *label_paths, iftFuzzyModelExtended *fuzzyModel){
    iftImage *instance;
    iftVoxel cInstance, global_center = {0, 0, 0}, dMin, dMax, dAct;

    dMin.x = dMin.y = dMin.z = IFT_INFINITY_INT;
    dMax.x = dMax.y = dMax.z = IFT_INFINITY_INT_NEG;

    // Center of the Model on the Ref. Space
    // global_center = fuzzyModel->ref_info->model_center;

    for(int j = 0; j < label_paths->n; j++) {
        instance = iftReadImage(label_paths->files[j]->path);

        // cInstance = iftGeometricCenterVoxel(instance);
        iftBoundingBox mbb = iftMinBoundingBox(instance, NULL);
        // Center of the Object Bounding Box on the Ref. Space
        cInstance.x = (mbb.begin.x + mbb.end.x) / 2;
        cInstance.y = (mbb.begin.y + mbb.end.y) / 2;
        cInstance.z = (mbb.begin.z + mbb.end.z) / 2;

        // finds out the Search Region
        dAct.x = cInstance.x - global_center.x;
        dAct.y = cInstance.y - global_center.y;
        dAct.z = cInstance.z - global_center.z;
        dMin.x = iftMin(dAct.x, dMin.x);
        dMin.y = iftMin(dAct.y, dMin.y);
        dMin.z = iftMin(dAct.z, dMin.z);
        dMax.x = iftMax(dAct.x, dMax.x);
        dMax.y = iftMax(dAct.y, dMax.y);
        dMax.z = iftMax(dAct.z, dMax.z);

        iftDestroyImage(&instance);
    }

    fuzzyModel->start = dMin;
    fuzzyModel->end = dMax;
}


// float iftGetMeanScale(char *binsRoute){

//    fileList *binFiles;
//    iftDataSet *Z, *Zc;
//    iftMatrix *U,*S,*Vt,*A;
//    float scale, scaleSum = 0, meanScale;
//    iftImage *bin;

//    binFiles = iftGetFiles(binsRoute, ".scn");
//    for(int i = 0; i < binFiles->n; i++) {
//       bin = iftReadImage(binFiles->filesRoutes[i]);
//       Z   = iftObjectToDataSet(bin);
//       iftSetStatus(Z,IFT_TRAIN);
//       Zc  = iftCentralizeDataSet(Z);
//       A = iftDatasetCovarianceMatrix(Zc);
//       iftSingleValueDecomp(A,&U,&S,&Vt);
//       scale = sqrt(S->val[0]*S->val[0]+S->val[1]*S->val[1]+S->val[2]*S->val[2]);
//       scaleSum += scale;
//       iftDestroyDataSet(&Z);
//       iftDestroyDataSet(&Zc);
//       iftDestroyMatrix(&A);
//       iftDestroyMatrix(&U);
//       iftDestroyMatrix(&S);
//       iftDestroyMatrix(&Vt);
//       iftDestroyImage(&bin);   
//    }
//    meanScale = scaleSum / binFiles->n;
//    iftDestroyFileList(&binFiles);
//    return meanScale;
// }

void iftWriteStatisticModelSimple(iftStatisticModelSimple *statisticModel, char *filename){
    FILE       *fp=NULL;

    fp = fopen(filename,"wb");
    if (fp == NULL){
        iftError(MSG_FILE_OPEN_ERROR, "iftWriteStatisticModelSimple");
    }

    fprintf(fp,"%5.4f %5.4f %d\n",statisticModel->mean,statisticModel->deviation,statisticModel->mode);

    fprintf(fp,"FSCN\n");
    fprintf(fp,"%d %d %d\n",statisticModel->model->xsize,statisticModel->model->ysize,statisticModel->model->zsize);
    fprintf(fp,"%f %f %f\n",statisticModel->model->dx,statisticModel->model->dy,statisticModel->model->dz);
    fwrite(statisticModel->model->val,sizeof(float),statisticModel->model->n,fp);

    fclose(fp);
}

void iftDestroyStatisticModelSimple(iftStatisticModelSimple **statisticModel){
    if (statisticModel != NULL){
        if ((*statisticModel)->model != NULL)
            iftDestroyFImage(&((*statisticModel)->model));
        *statisticModel = NULL;
    }
}

iftStatisticModelSimple *iftReadStatisticModelSimple(char *filename){
    iftStatisticModelSimple *statisticModel;
    iftFImage  *img=NULL;
    FILE       *fp=NULL;
    char        type[10];
    int         xsize,ysize,zsize;
    long        pos;

    fp = fopen(filename,"rb");
    if (fp == NULL){
        iftError(MSG_FILE_OPEN_ERROR, "iftReadStatisticModelSimple");
    }

    statisticModel = (iftStatisticModelSimple *) iftAlloc(1,sizeof(iftStatisticModelSimple));

    if (fscanf(fp,"%f %f %d",&(statisticModel->mean),&(statisticModel->deviation),&(statisticModel->mode))!=3)
        iftError("Reading error","iftReadStatisticModelSimple Parameters");

    if (fscanf(fp,"%s",type)!=1) iftError("Reading error","iftReadStatisticModelSimple Model");
    if((strcmp(type,"FSCN")==0)){
        if (fscanf(fp,"%d %d %d",&xsize,&ysize,&zsize)!=3) iftError("Reading error","iftReadStatisticModelSimple Model");
        img = iftCreateFImage(xsize,ysize,zsize);
        if (fscanf(fp,"%f %f %f",&img->dx,&img->dy,&img->dz)!=3) iftError("Reading error","iftReadStatisticModelSimple Model");
        pos = ftell(fp);
        fseek(fp,(pos+1)*sizeof(char),SEEK_SET);
        if(fread(img->val,sizeof(float),img->n,fp)!=img->n) iftError("Reading error","iftReadStatisticModelSimple Model");
        statisticModel->model = img;
    }else{
        iftError("Invalid model type","iftReadStatisticModelSimple Model");
    }

    fclose(fp);
    return(statisticModel);

}

// void iftComputeHistogramParameters(char *dirImages, char *dirInstances, float *meanP, int *modeP, float *sdP)
// {
//    fileList *imageFiles, *segFiles; 
//    iftImage *image, *segmentation;
//    int histogram [4096];
//    float total, count, squareSum, mean;
//    int mode;

//    for (int i = 0; i < 4096; i++) histogram[i] = 0;

//    imageFiles = iftGetFiles(dirImages, ".scn");
//    segFiles = iftGetFiles(dirInstances, ".scn");

//    for(int i = 0; i < imageFiles->n; i++) 
//    {
//       for(int j = 0; j < segFiles->n; j++) 
//          if (strcmp(strstr(imageFiles->filesNames[i], "_"), strstr(segFiles->filesNames[j], "_"))==0)
//          {
//             image = iftReadImage(imageFiles->filesRoutes[i]);
//             segmentation = iftReadImage(segFiles->filesRoutes[j]);
//             for (int k = 0; k < image->n; k++) 
//                if (segmentation->val[k] > 0)
//                   histogram[image->val[k]] += 1;            
//                iftDestroyImage(&image);
//                iftDestroyImage(&segmentation);
//          }
//    }

//    //Calculate mean and sd
//    mode = 100;
//    total = 0;
//    count = 0;
//    squareSum = 0;
//    for (int i = 5; i < 4096; i++)
//    {
//       total += (histogram[i] * i);
//       count += histogram[i];
//       if (histogram[i] > histogram[mode]) 
//          mode = i;
//    }
//    mean = total / count;
//    for (int i = 5; i < 4096; i++)
//       squareSum += (histogram[i] * (i - mean) * (i - mean));

//    *meanP = mean;
//    *sdP = sqrt(squareSum / count);   
//    *modeP = mode;

//    iftDestroyFileList(&imageFiles);
//    iftDestroyFileList(&segFiles);
// }

// void iftFuzzyByAveragingNoGCCentering(iftFImage *model, fileList *imageFiles){

//    iftImage *image, *croppedImage;
//    iftVoxel cropVoxel, modelVoxel;
//    int pCrop, pModel;

//    for(int i = 0; i < imageFiles->n; i++) {
//       image = iftReadImage(imageFiles->filesRoutes[i]);
//       iftBoundingBox mbb = iftMinObjectBoundingBox(image, 1, NULL);
//       croppedImage            = iftExtractROI(image, mbb);

//       for (cropVoxel.z=0; cropVoxel.z < croppedImage->zsize; cropVoxel.z++) 
//          for (cropVoxel.y=0; cropVoxel.y < croppedImage->ysize; cropVoxel.y++) 
//             for (cropVoxel.x=0; cropVoxel.x < croppedImage->xsize; cropVoxel.x++) {
//            pCrop = iftGetVoxelIndex(croppedImage, cropVoxel);
//                modelVoxel.x = cropVoxel.x;
//                modelVoxel.y = cropVoxel.y;
//                modelVoxel.z = cropVoxel.z;
//                pModel = iftGetVoxelIndex(model, modelVoxel);
//                if (iftFValidVoxel(model, modelVoxel))
//                   model->val[pModel] = model->val[pModel] + croppedImage->val[pCrop];                  
//             }
//       iftDestroyImage(&image);
//       iftDestroyImage(&croppedImage); 
//    }

//    for (int k=0; k < model->n; k++)
//       model->val[k] = model->val[k] / imageFiles->n;

// }

void iftModelIntensitiesFuzzyExtended(iftFileSet *img_paths, iftFileSet *label_paths, iftFuzzyModelExtended *fuzzyModel){

    iftImage *image, *segmentation;
    int histogram [4096];
    float total, count, squareSum;
    int mode;

    for (int i = 0; i < 4096; i++) histogram[i] = 0;

    for(int i = 0; i < img_paths->n; i++)
    {
        for(int j = 0; j < label_paths->n; j++) {

            if ((img_paths->files[i]->label == label_paths->files[j]->label) &&
                (img_paths->files[i]->sample == label_paths->files[j]->sample))
            {
                image = iftReadImage(img_paths->files[i]->path);
                segmentation = iftReadImage(label_paths->files[j]->path);
                for (int k = 0; k < image->n; k++)
                    if (segmentation->val[k] > 0)
                    {
                        histogram[image->val[k]] += 1;
                    }
                iftDestroyImage(&image);
                iftDestroyImage(&segmentation);
            }
        }
    }

    //Calculate mean and sd
    mode = 100;
    total = 0;
    count = 0;
    squareSum = 0;
    for (int i = 5; i < 4096; i++)
    {
        total += (histogram[i] * i);
        count += histogram[i];
        if (histogram[i] > histogram[mode])
            mode = i;
    }
    fuzzyModel->mean = total / count;
    for (int i = 5; i < 4096; i++)
        squareSum += (histogram[i] * (i - fuzzyModel->mean) * (i - fuzzyModel->mean));

    fuzzyModel->deviation = sqrt(squareSum / count);
    fuzzyModel->mode = mode;
}

//Computes the fuzzy model, by averaging the sum of pixels of every segmented image
void iftFuzzyByAveragingDilated(iftFImage *model, iftFileSet *img_paths, int obj_label) {

    // iftImage *image, *croppedImage;
    iftImage *image;
    iftVoxel centerCrop, centerModel, translation, cropVoxel, modelVoxel;
    int pCrop, pModel;

    centerModel.x = model->xsize / 2;
    centerModel.y = model->ysize / 2;
    centerModel.z = model->zsize / 2;

    for(int i = 0; i < img_paths->n; i++) {
        image = iftReadImage(img_paths->files[i]->path);
        iftBoundingBox mbb = iftMinObjectBoundingBox(image, obj_label, NULL);
        // croppedImage       = iftExtractROI(image, mbb);
        // centerCrop = iftGeometricCenterVoxel(croppedImage);
        centerCrop.x = (mbb.begin.x + mbb.end.x) / 2;
        centerCrop.y = (mbb.begin.y + mbb.end.y) / 2;
        centerCrop.z = (mbb.begin.z + mbb.end.z) / 2;
        translation.x = centerModel.x - centerCrop.x;
        translation.y = centerModel.y - centerCrop.y;
        translation.z = centerModel.z - centerCrop.z;

        for (cropVoxel.z=mbb.begin.z; cropVoxel.z <= mbb.end.z; cropVoxel.z++)
            for (cropVoxel.y=mbb.begin.y; cropVoxel.y <= mbb.end.y; cropVoxel.y++)
                for (cropVoxel.x=mbb.begin.x; cropVoxel.x <= mbb.end.x; cropVoxel.x++) {
                    pCrop = iftGetVoxelIndex(image, cropVoxel);
                    modelVoxel.x = cropVoxel.x + translation.x;
                    modelVoxel.y = cropVoxel.y + translation.y;
                    modelVoxel.z = cropVoxel.z + translation.z;
                    pModel = iftGetVoxelIndex(model, modelVoxel);
                    if ((iftFValidVoxel(model, modelVoxel)) && (image->val[pCrop] == obj_label))
                        model->val[pModel] = model->val[pModel] + 1;
                }
        iftDestroyImage(&image);
        // iftDestroyImage(&croppedImage);
    }

    for (int k=0; k < model->n; k++)
        model->val[k] = model->val[k] / img_paths->n;
}

// iftStatisticModelSimple *iftMakeStatisticModelSimple(char *imagesDirectory, char *segmentationsDirectory)
// {
//    iftStatisticModelSimple *statisticModel;
//    fileList *imageFiles, *segFiles;
//    iftImage *image, *segmentation;
//    int histogram [4096];
//    float total, count, squareSum;
//    int mode;

//    for (int i = 0; i < 4096; i++) histogram[i] = 0;
//    statisticModel = (iftStatisticModelSimple *) iftAlloc(1,sizeof(iftStatisticModelSimple));
//    imageFiles = iftGetFiles(imagesDirectory, ".scn");
//    segFiles = iftGetFiles(segmentationsDirectory, ".scn");
//    image = iftReadImage(imageFiles->filesRoutes[0]);
//    statisticModel->model = iftCreateFImage(image->xsize, image->ysize, image->zsize);
//    iftDestroyImage(&image);

//    for (int k = 0; k < statisticModel->model->n; k++)
//       statisticModel->model->val[k] = 0;

//    for(int i = 0; i < imageFiles->n; i++)
//    {
//       for(int j = 0; j < segFiles->n; j++)
//          if (strcmp(strstr(imageFiles->filesNames[i], "_"), strstr(segFiles->filesNames[j], "_"))==0)
//          {
//             image = iftReadImage(imageFiles->filesRoutes[i]);
//             segmentation = iftReadImage(segFiles->filesRoutes[j]);
//             for (int k = 0; k < statisticModel->model->n; k++)
//                if (segmentation->val[k] > 0)
//                {
//                   statisticModel->model->val[k] += 1;
//                   histogram[image->val[k]] += 1;
//                }
//             iftDestroyImage(&image);
//             iftDestroyImage(&segmentation);
//          }
//    }

//    for (int k = 0; k < statisticModel->model->n; k++)
//       statisticModel->model->val[k] = statisticModel->model->val[k] / imageFiles->n;

//    //Calculate mean and sd
//    mode = 100;
//    total = 0;
//    count = 0;
//    squareSum = 0;
//    for (int i = 5; i < 4096; i++)
//    {
//       total += (histogram[i] * i);
//       count += histogram[i];
//       if (histogram[i] > histogram[mode])
//          mode = i;
//    }
//    statisticModel->mean = total / count;
//    for (int i = 5; i < 4096; i++)
//       squareSum += (histogram[i] * (i - statisticModel->mean) * (i - statisticModel->mean));

//    statisticModel->deviation = sqrt(squareSum / count);
//    statisticModel->mode = mode;

//    iftDestroyFileList(&imageFiles);
//    iftDestroyFileList(&segFiles);

//    return statisticModel;
// }

iftImage *iftSegmentWithStatisticModelSimple(char *imageRoute, char *modelRoute)
{
    iftImage *image, *result;
    iftStatisticModelSimple *model;
    float threshold = 0.5, gaussianValue, aposteriori, firstVal, secondVal, amplitude = 0.5;
    int histogram [4096];

    image = iftReadImage(imageRoute);
    for (int i = 0; i < 4096; i++)
        histogram[i] = 0;
    for (int i = 0; i < image->n; i++)
        histogram[image->val[i]] += 1;

    model = iftReadStatisticModelSimple(modelRoute);
    result = iftCreateImage(image->xsize, image->ysize, image->zsize);

    for (int i = 0; i < image->n; i++)
    {
        firstVal = 0.5 * (1 + erf(( (image->val[i] - amplitude) - model->mode) / (model->deviation * sqrt(2.))));
        secondVal = 0.5 * (1 + erf(( (image->val[i] + amplitude) - model->mode) / (model->deviation * sqrt(2.))));
        gaussianValue = secondVal - firstVal;
        aposteriori = model->model->val[i] * gaussianValue / ((float)histogram[image->val[i]] / image->n);
        if (aposteriori > threshold)
            result->val[i] = 1;
        else
            result->val[i] = 0;
    }

    iftDestroyImage(&image);
    iftDestroyStatisticModelSimple(&model);

    return result;
}

iftImage* iftAPosterioriGradientOld(iftImage *image, iftFuzzyModelExtended *model, float alpha)
{
    iftImage *resultProbabilityGradient, *resultGradient, *resultFinal;
    iftFImage *result;
    float gaussianValue, firstVal, secondVal, amplitude = 0.5, p;
    iftAdjRel *adjacency;

    adjacency = iftSpheric(sqrtf(1.0));
    p = alpha;

    result = iftCreateFImage(image->xsize, image->ysize, image->zsize);
    resultFinal = iftCreateImage(image->xsize, image->ysize, image->zsize);
    for (int i = 0; i < image->n; i++)
    {
        firstVal = 0.5 * (1 + erf(( (image->val[i] - amplitude) - model->mode) / (model->deviation * sqrt(2.))));
        secondVal = 0.5 * (1 + erf(( (image->val[i] + amplitude) - model->mode) / (model->deviation * sqrt(2.))));
        gaussianValue = secondVal - firstVal;
        result->val[i] = gaussianValue;
    }

    resultProbabilityGradient = iftImageGradientMagnitude(iftFImageToImage(result, 1024), adjacency);
    resultGradient = iftImageGradientMagnitude(image, adjacency);

    for (int i = 0; i < image->n; i++)
        resultFinal->val[i] = iftRound((resultGradient->val[i]) * p + (resultProbabilityGradient->val[i]) * (1 - p) );

    iftDestroyFImage(&result);
    iftDestroyImage(&resultGradient);
    iftDestroyImage(&resultProbabilityGradient);
    iftDestroyAdjRel(&adjacency);

    return resultFinal;
}


iftMatrix *iftObjectAlignMatrixByPCA(iftImage *bin)
{
    iftDataSet *Z=iftObjectToDataSet(bin);

    iftSetStatus(Z,IFT_TRAIN);
    iftDataSet *Zc=iftCentralizeDataSet(Z);
    iftMatrix  *C = iftDatasetCovarianceMatrix(Zc);
    iftMatrix  *U,*S,*Vt;
    iftSingleValueDecomp(C,&U,&S,&Vt);
    iftDestroyMatrix(&C);
    iftDestroyMatrix(&U);
    iftDestroyMatrix(&S);
    iftDestroyDataSet(&Zc);
    iftDestroyDataSet(&Z);

    /* It assumes that the alignment must not flip the object along any
       axis. */

    for (int i=0; i < Vt->nrows; i++)
        if (Vt->val[iftGetMatrixIndex(Vt,i,i)] < 0)
            Vt->val[iftGetMatrixIndex(Vt,i,i)]=-Vt->val[iftGetMatrixIndex(Vt,i,i)];

    return(Vt);
}

void iftImageSetVoxelIntensityStatistics(  iftFileSet *img_files,   iftFileSet *mask_files,
                                         double *intensities_mode, double *intensities_mode_deviation,
                                         int normalization_value) {
    int i;
    iftImage *img = NULL, *label = NULL, *norm_img = NULL;
    iftHist *hist = NULL, *tmp = NULL, *cur_hist = NULL;

    if(img_files->n != mask_files->n)
        iftError("The image file set and mask file set must refer to the same images and have the same sizes",
                 "iftImageSetVoxelIntensityStatistics");

    for(i = 0; i < img_files->n; i++) {
        img = iftReadImageByExt(img_files->files[i]->path);
        label = iftReadImageByExt(mask_files->files[i]->path);

        if(hist == NULL) {
            // Initializing the histogram by computing it for the first image
//            normalization_value = iftNormalizationValue(iftMaximumValue(img));
            hist = iftGrayHistogramInRegion(img, label, normalization_value+1, 0);
        } else {
            // Normalizing images according to the normalization value
            norm_img = iftLinearStretch(img, iftMinimumValue(img), iftMaximumValue(img), 0, normalization_value);

//            cur_normalization_value = iftNormalizationValue(iftMaximumValue(norm_img));

            // Computing the histogram for the new image
            cur_hist = iftGrayHistogramInRegion(norm_img, label, hist->nbins, 0);

            // Adding both histograms
            tmp = iftAddHistograms(hist, cur_hist);

            iftDestroyHist(&hist);
            iftDestroyHist(&cur_hist);
            hist = tmp;
        }

        iftDestroyImage(&img);
        iftDestroyImage(&norm_img);
        iftDestroyImage(&label);
    }

    // Computing the mode of the combined histogram
    *intensities_mode = iftHistogramMode(hist, 0);

    // Computing the deviation to the mode
    *intensities_mode_deviation = iftHistogramStdevAroundValue(hist, *intensities_mode, 0);

    iftDestroyHist(&hist);
}


iftFImage *iftEnhanceImageByGaussian(iftImage *image, double value, double deviation) {
    iftFImage *enhanced = NULL;

    enhanced = iftCreateFImage(image->xsize, image->ysize, image->zsize);

#pragma omp parallel for
    for (int i = 0; i < image->n; i++) {
        double gaussianValue, firstVal, secondVal, amplitude = 0.5;

        firstVal = 0.5 * (1 + erf(((image->val[i] - amplitude) - value) / (deviation * sqrt(2.))));
        secondVal = 0.5 * (1 + erf(((image->val[i] + amplitude) - value) / (deviation * sqrt(2.))));
        gaussianValue = secondVal - firstVal;
        enhanced->val[i] = gaussianValue;
    }

    iftFCopyVoxelSizeFromImage(image, enhanced);

    return enhanced;
}

iftImage *iftGaussianEnhancedImageGradient(iftImage *image, iftAdjRel *A, double value, double deviation, double alpha) {
    iftImage *grad_img = NULL;
    iftFImage *enhanced = NULL;
    iftImage *enhanced_img = NULL;
    iftImage *grad_enhanced = NULL;
    iftImage *grad_combined = NULL;
    iftImage *tmp = NULL;

    if(alpha < 0.0 || alpha > 1.0)
        iftError("The linear combination weight must be within [0.0, 1.0] and not %lf!",
                 "iftGaussianEnhancedImageGradient", alpha);

    grad_img = iftImageGradientMagnitude(image, A);

    /* Enhancing the object if alpha is not 0.0*/
    if(!iftAlmostZero(alpha - 0.0)) {
        enhanced = iftEnhanceImageByGaussian(image, value, deviation);
        enhanced_img = iftFImageToImage(enhanced, IFT_MAXWEIGHT);
        grad_enhanced = iftImageGradientMagnitude(enhanced_img, A);
        iftDestroyFImage(&enhanced);
        iftDestroyImage(&enhanced_img);

        tmp = iftLinearStretch(grad_enhanced, iftMinimumValue(grad_enhanced), iftMaximumValue(grad_enhanced), 0,
                               IFT_MAXWEIGHT);
        iftDestroyImage(&grad_enhanced);
        grad_enhanced = tmp;

        /* Combining the enhanced gradient with the image's gradient if alpha is not 1.0 */
        if(!iftAlmostZero(alpha - 1.0)) {
            tmp = iftLinearStretch(grad_img, iftMinimumValue(grad_img), iftMaximumValue(grad_img), 0, IFT_MAXWEIGHT);
            iftDestroyImage(&grad_img);
            grad_img = tmp;

            grad_combined = iftLinearCombination(grad_enhanced, grad_img, alpha);
            iftDestroyImage(&grad_enhanced);
        } else {
            /* Copying the enhanced gradient if alpha is 1.0 */
            grad_combined = grad_enhanced;
        }
    } else {
        tmp = iftLinearStretch(grad_img, iftMinimumValue(grad_img), iftMaximumValue(grad_img), 0, IFT_MAXWEIGHT);
        iftDestroyImage(&grad_img);

        /* Copying the image's gradient if alpha is 1.0 */
        grad_combined = tmp;
    }

    if(grad_combined != grad_img) iftDestroyImage(&grad_img);

    return grad_combined;
}

/** MSPS-based optimum object search by translation **/


/** MSPS-based optimum object search by translation **/















float iftMSPSObjectModelSearchByTranslationSingleCenterProblem(void *problem, float *theta) {
    int p;
    double score = 0.0;
    iftObjectModelProb *prob = (iftObjectModelProb *)problem;
    iftVoxel cur_pos[1];
    iftLabeledSet *objects = NULL, *borders = NULL;

    // Initializing the score as the worst possible one, which depends on whether we are
    // maximizing or minimizing the recognition functional
    score = prob->recognition_data->worst_score;

    // Obtaining the current position of the model
    cur_pos[0].x = iftRound(theta[0]);
    cur_pos[0].y = iftRound(theta[1]);
    cur_pos[0].z = 0;

    // Getting the Z coordinate for 3D images
    if(iftIs3DImage(prob->delineation_data->orig))
        cur_pos[0].z = iftRound(theta[2]);

    // Only evaluate the objective function in valid search voxels
    if(iftValidVoxel(prob->delineation_data->orig, cur_pos[0])) {
        p = iftGetVoxelIndex(prob->delineation_data->orig, cur_pos[0]);

        if(iftBMapValue(prob->search_region, p)) {
            // Prearing the recognition algorithm if necessary
            if(prob->iftRecogAlgIterationSetup != NULL)
                prob->iftRecogAlgIterationSetup(prob, cur_pos);

            // Delineating object with model
            iftMSPSDelineateObjectWithModel(prob, cur_pos, &objects, &borders);

            // Performing post-processing after delineation when required by the recognition algorithm. Post-processing
            // may fill the whole in segmentation caused by running the delineation algorithm only inside the uncertainty
            // region, for instance.
            if (prob->iftDelinAlgPostProcessing != NULL)
                prob->iftDelinAlgPostProcessing(prob->model, prob->recognition_data, &objects, &borders);

            // Running the recognition algorithm
            score = prob->iftRecogAlgorithm(prob, objects, borders);

            // Resetting the delineation algorithm. IMPORTANT: this effectively erases the current delineation result and
            // must be done after the recognition algorithm is called to ensure data consistency
            if (prob->iftDelinAlgReset != NULL)
                prob->iftDelinAlgReset(prob->delineation_data, objects);

            // Cleaning up!
            iftDestroyLabeledSet(&objects);
            iftDestroyLabeledSet(&borders);
        }
    }

    return score;
}

void iftMSPSDelineateObjectWithModel(iftObjectModelProb *prob, iftVoxel *cur_pos, iftLabeledSet **objects,
                                     iftLabeledSet **borders) {
    iftLabeledSet *S = NULL;
    iftSet *Uncertain = NULL;

    // Preparing the delineation algorithm (i.e., obtaining placing the seeds on the target image)
    prob->iftDelinAlgIterationSetup(prob, cur_pos, &S, &Uncertain);
    // Delineating the target image
    prob->iftDelinAlgorithm(prob->delineation_data, S, Uncertain, objects, borders);

    // Cleaning up!
    iftDestroyLabeledSet(&S);
    iftDestroySet(&Uncertain);
}


// Initializes the MSPS parameters
void iftInitializeMSPSObjectModelSearchByTranslationParamsSingleCenter(iftMSPS *msps, int min_disp,
                                                                       int scale_disp_increment) {
    // MSPS translation deltas in X, Y, and Z
    for (int i = 0; i < msps->n; i++) {
        msps->delta->val[iftGetMatrixIndex(msps->delta,i,0)] = min_disp;
        msps->sigma->val[iftGetMatrixIndex(msps->delta,i,0)] = iftMax(scale_disp_increment/2, 1);
        for(int j = 1; j < msps->m; j++) {
            msps->delta->val[iftGetMatrixIndex(msps->delta,i,j)] += msps->delta->val[iftGetMatrixIndex(msps->delta,i,j-1)] + scale_disp_increment;
            msps->sigma->val[iftGetMatrixIndex(msps->delta,i,j)] = iftMax(scale_disp_increment/2, 1);
        }
    }
}


iftMSPS *iftMSPSObjectModelSearchByTranslationSingleCenterInit(iftObjectModelProb *prob, int nscales,
                                                               int min_disp, int scale_disp_increment) {
    int ndim = (iftIs3DImage(prob->delineation_data->orig)) ? 3 : 2;
    iftMSPS *msps = NULL;

    // Allocating the MSPS data
    msps = iftCreateMSPS(ndim, nscales, iftMSPSObjectModelSearchByTranslationSingleCenterProblem, prob);
    // Initializing the MSPS search parameters
    iftInitializeMSPSObjectModelSearchByTranslationParamsSingleCenter(msps, min_disp, scale_disp_increment);

    // Selecting the perturbation function
    msps->iftPerturbation = iftMSPSLinearRandomPerturbation;

    // Initializing the delineation algorithm data if necessary
    if(prob->iftDelinAlgInit != NULL)
        prob->iftDelinAlgInit(prob->delineation_data);

    return msps;
}

float iftMSPSObjectModelSearchByTranslationSingleCenterOptimization(iftMSPS *msps, iftVoxel start, iftVoxel *best_pos) {

    float best_score;
    iftObjectModelProb *prob = (iftObjectModelProb*)msps->problem;

    // IMPORTANT: We assume that the model has already been mapped onto
    // the initial position in the target image
    msps->theta[0]     = start.x;
    msps->theta[1]     = start.y;

    if(msps->n > 2)
        msps->theta[2] = start.z;

    // Optimizing!!
    if (prob->recognition_data->opt_str == IFT_MAXIMIZATION)
        best_score = iftMSPSMax(msps);
        // Minimization
    else
        best_score = iftMSPSMin(msps);

    // Returning the best position for the model
    best_pos->x = iftRound(msps->theta[0]);
    best_pos->y = iftRound(msps->theta[1]);
    best_pos->z = (msps->n > 2) ? iftRound(msps->theta[2]) : 0; // If the model is 3D, then we return the Z position as well

    return best_score;
}




void iftOMDelinWatershedInit(iftOMDelinAlgData *data) {
    iftOMDelinAlgWatershedCustomData * custom_data = NULL;

    custom_data = (iftOMDelinAlgWatershedCustomData *)data->custom_data;
    iftResetImageForest(custom_data->fst);

    // Setting the cost of all voxels to -infinity to ensure that IFT-SC
    // only occurs inside the uncertainty region
    iftSetImage(custom_data->fst->pathval, IFT_INFINITY_INT_NEG);
}


void iftOMDelinWatershedIterationSetupSingleCenter(iftObjectModelProb *prob, iftVoxel *cur_pos, iftLabeledSet **S,
                                                   iftSet **Uncertain) {
    size_t j;
    int i, p;
    iftVoxel u;
    iftOMDelinAlgWatershedCustomData * custom_data = prob->delineation_data->custom_data;
    // iftBMap *restricted_region = custom_data->mask;

    if(S == NULL)
        iftError("The seed labeled set S must not be NULL!", "iftOMDelinWatershedIterationSetupSingleCenter");
    if(Uncertain == NULL)
        iftError("The uncertainty region set Uncertain must not be NULL!", "iftOMDelinWatershedIterationSetupSingleCenter");

    // if(restricted_region == NULL)
    //     iftError("The delineation data must contain a non-NULL restricted region!", "iftOMDelinWatershedIterationSetupSingleCenter");

    *S = NULL;
    *Uncertain = NULL;

    // Selecting valid uncertainty region voxels only inside the restricted region
    for (i = 0; i < prob->uncertainty_adj->n; i++) {
        u = iftGetAdjacentVoxel(prob->uncertainty_adj, cur_pos[0], i);
        if (iftValidVoxel(custom_data->fst->img, u)) {
            p = iftGetVoxelIndex(custom_data->fst->img, u);
            iftInsertSet(Uncertain, p);
        }
    }

    // Selecting valid object seed voxels only inside the restricted region
    for(j = 0; j < prob->labels->n; j++) {
        for (i = 0; i < prob->seed_adj[j]->n; i++) {
            u = iftGetAdjacentVoxel(prob->seed_adj[j], cur_pos[0], i);
            if (iftValidVoxel(custom_data->fst->img, u)) {
                p = iftGetVoxelIndex(custom_data->fst->img, u);
                iftInsertLabeledSet(S, p, prob->labels->val[j]);
            }
        }
    }
}

void iftOMDelinWatershed(iftOMDelinAlgData *data, iftLabeledSet *seeds,
                         iftSet *uncertainty, iftLabeledSet **objects, iftLabeledSet **borders) {
    int i, p, q, tmp;
    iftVoxel u, v;
    iftOMDelinAlgWatershedCustomData * custom_data = (iftOMDelinAlgWatershedCustomData *)data->custom_data;
    iftImageForest *fst = custom_data->fst;
    iftImage *basins = fst->img, *pathval = fst->pathval;
    iftImage *label = fst->label, *pred = fst->pred, *root = fst->root;
    iftGQueue *Q = fst->Q;
    iftAdjRel *A = fst->A;
    iftLabeledSet *aux = NULL;
    iftSet *aux_uncertain = NULL;
    iftBMap *border_processed = custom_data->border_processed;

    // Resetting the processed/labeled voxels and setting the variable to NULL
    if(objects != NULL)
        iftDestroyLabeledSet(objects);

    if (borders != NULL)
        iftDestroyLabeledSet(borders);

    // Setting costs for voxels in the uncertainty region
    for (aux_uncertain = uncertainty; aux_uncertain != NULL; aux_uncertain = aux_uncertain->next) {
        p = aux_uncertain->elem;
        pathval->val[p] = IFT_INFINITY_INT;
        pred->val[p] = IFT_NIL;
        root->val[p] = p;
    }

    // Setting costs and labels for seed voxels
    for (aux = seeds; aux != NULL; aux = aux->next) {
        p = aux->elem;

        label->val[p] = aux->label;
        pathval->val[p] = 0;
        root->val[p] = p;
        pred->val[p] = IFT_NIL;
        iftInsertGQueue(&Q, p);
    }

    while (!iftEmptyGQueue(Q)) {
        p = iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(basins, p);

        // Saving processed voxels
        if(objects != NULL) iftInsertLabeledSet(objects, p, label->val[p]);

        for (i = 1; i < A->n; i++) {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(basins, v)) {
                q = iftGetVoxelIndex(basins, v);
                if (pathval->val[q] > pathval->val[p]) {
                    tmp = iftMax(pathval->val[p], basins->val[q]);
                    if (tmp < pathval->val[q]) { // For this path-value function, this implies that q has never been inserted in Q.
                        label->val[q] = label->val[p];
                        pathval->val[q] = tmp;
                        pred->val[q] = p;
                        root->val[q] = root->val[p];
                        iftInsertGQueue(&Q, q);
                    }
                } else if (borders != NULL) {
                    // Computing the object's boundary if it was passed as a parameter and returning it
                    if ((Q->L.elem[q].color == IFT_BLACK) && (label->val[p] != label->val[q]) &&
                        !iftBMapValue(border_processed, p)) { /*  p and q must be in the Frontier set */
                        // Inserting p as a border voxel
                        iftInsertLabeledSet(borders, p, label->val[p]);
                        iftBMapSet1(border_processed, p);

                        if (!iftBMapValue(border_processed, q)) { // Inserting q as a border voxel if it wasn't before
                            iftInsertLabeledSet(borders, q, label->val[q]);
                            iftBMapSet1(border_processed, q);
                        }
                    }
                }
            }
        }
    }
}



void iftOMDelinWatershedReset(iftOMDelinAlgData *data, iftLabeledSet *processed) {
    iftLabeledSet *aux = NULL;
    int i, p;
    iftOMDelinAlgWatershedCustomData *custom_data = data->custom_data;
    iftBMap *border_processed = custom_data->border_processed;


    custom_data->fst->Q->C.minvalue = IFT_INFINITY_INT;
    custom_data->fst->Q->C.maxvalue = IFT_INFINITY_INT_NEG;

    for (i=0; i < custom_data->fst->Q->C.nbuckets+1; i++)
        custom_data->fst->Q->C.first[i] = custom_data->fst->Q->C.last[i]=IFT_NIL;
    for (aux = processed; aux != NULL; aux = aux->next) {
        p = aux->elem;
        custom_data->fst->Q->L.elem[p].color = IFT_WHITE;
        custom_data->fst->Q->L.elem[p].next  =  custom_data->fst->Q->L.elem[p].prev = IFT_NIL;
        custom_data->fst->pathval->val[p]    = IFT_INFINITY_INT_NEG;
        custom_data->fst->label->val[p]      = 0;
        custom_data->fst->pred->val[p]       = IFT_NIL;
        custom_data->fst->root->val[p]       = p;
        iftBMapSet0(border_processed, p);
    }
}


float iftOMRecogAlgWatershedMeanCut(iftObjectModelProb *prob, iftLabeledSet *processed,
                                    iftLabeledSet *borders) {
    double score = 0.0, n = 0.0;
    iftLabeledSet *aux = NULL;
    iftOMDelinAlgWatershedCustomData *custom_data = prob->delineation_data->custom_data;
    iftImage *basins = custom_data->fst->img;

    for(aux = borders; aux != NULL; aux = aux->next) {
        score += basins->val[aux->elem];
        n++;
    }

    return score / iftMax(n, 1.0);
}