/**
 * @file
 * @brief Builds the Movable and Adaptive Probabilistic Atlas MAPA.
 * @note See the source code in @ref iftBuildMAPA.c
 *
 * @example iftBuildMAPA.c
 * @brief Builds the Movable and Adaptive Probabilistic Atlas MAPA.
 * @author Samuel Martins
 * @date Jan 5, 2017
 */



#include "ift.h"
#include "ift/medical/segm/AdaPro.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, iftFileSet **atlas_set, char **template_path, char **out_adapro_path);
void iftValidateRequiredArgs(const char *atlas_entry, const char *template_path,
                             const char *out_adapro_path);
void iftGetOptionalArgs(  iftDict *args, const char *label_path, char **train_voxels_img_path,
                        iftIntArray **labels, iftFloatArray **e_radius_arr, iftFloatArray **d_radius_arr, double *C,
                        iftFileSet **elastix_files);
iftIntArray *iftGetLabelArrayFromCmdLine(  iftDict *args, const char *label_path);
iftFloatArray *iftParseRadiusEntry(const char *radius_entry,   iftIntArray *labels);
void iftGetRadius(  iftDict *args,   iftIntArray *labels, iftFloatArray **e_radius_arr,
                  iftFloatArray **d_radius_arr);
/*************************************************************/

int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    
    // mandatory args
    iftFileSet *atlas_set  = NULL;
    char *template_path    = NULL;
    char *out_adapro_path  = NULL;
    // optional args
    char *train_voxels_img_path = NULL;
    iftIntArray *labels = NULL;
    iftFloatArray *e_radius_arr; // erosion radius array
    iftFloatArray *d_radius_arr; // dilation radius array
    double C = 1e-5;
    iftFileSet *elastix_files = NULL;
    
    
    iftGetRequiredArgs(args, &atlas_set, &template_path, &out_adapro_path);
    iftGetOptionalArgs(args, atlas_set->files[0]->path, &train_voxels_img_path, &labels, &e_radius_arr, &d_radius_arr,
                       &C, &elastix_files);

    timer *t1 = iftTic();

    puts("- Reading Template");
    iftImage *template_img = iftReadImageByExt(iftAbsPathname(template_path));

    iftImage *train_voxels_img = NULL;
    if (train_voxels_img_path != NULL) {
        puts("- Reading Marker Image");
        train_voxels_img = iftReadImageByExt(iftAbsPathname(train_voxels_img_path));
    }

    puts("- Training Adaptive Prob. Atlas Bank");
    iftAdaPro *adapro = iftTrainAdaPro(atlas_set, template_img, train_voxels_img, labels, e_radius_arr, d_radius_arr, C,
                                       elastix_files);
    // // save the probabilistic atlases for visualization
    // for (int o = 0; o < adapro->labels->n; o++)
    //     iftWriteImageByExt(iftFImageToImage(iftProbAtlasOnTemplateImageDomain(adapro->obj_models[o]->prob_atlas, adapro->obj_models[o]->template_shape, adapro->obj_models[o]->begin), 4095),
    //                        "tmp/prob_atlas_%d.hdr", adapro->labels->val[o]);

    puts("- Writing Adaptive Prob. Atlas Bank");
    iftWriteAdaPro(adapro, out_adapro_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&atlas_set);
    iftDestroyImage(&template_img);
    iftDestroyImage(&train_voxels_img);
    iftDestroyAdaPro(&adapro);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Train an Adaptive Probabilistic Atlas (AdaProb).\n" \
        "- If the Target Object Labels are not passed, all Labels from the First Training Atlas will be considered " \
        "for the construction of the Models.\n" \
        "- For texture classifier, it trains a single linear SVM for all objects. The training samples " \
        "are labeled voxels (bg = 1, objects = 2) chosen from a label image marked on the template image.\n" \
        "- If no label image with the training voxels is passed (option --training-voxels), the model won't be adaptive, only relying on " \
        "shape constraints of the probabilistic atlases during segmentation.\n\n" \
        "- Optionally, it is possible to pass the erosion and dilation radius for each target object, " \
        "which will be used during the segmentation.";


    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-l", .long_name = "--atlas-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Dir with the Atlases (Label Images already registered) or a CSV file with their pathnames."},
        {.short_name = "-r", .long_name = "--template", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Template (Reference Image) used to register the Input Label Images."},
        {.short_name = "-o", .long_name = "--output-adapro", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Zip File to store the Adaptive Prob. Atlas adapro"},
        {.short_name = "-t", .long_name = "--training-voxels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Label Image with the training voxels (labeled voxels) which will be training samples for linear SVM."},
        {.short_name = "-j", .long_name = "--obj-labels", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Target Objects. Ex: 1,2,5 (3 objects)\nDefault: All labels from the First Image"},
        {.short_name = "-e", .long_name = "--erosion-radius_arr", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Set of Erosion Radius (one for each object or an overall radius_arr) " \
                                "to find the Inner Seeds from the Prob. Atlas (Certain Region Border of the Target Object).\n" \
                                 "Ex: -e 0,3,3 or -e 10\nDefault: 0"},
        {.short_name = "-d", .long_name = "--dilation-radius_arr", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Set of Dilation Radius (one for each object or an overall radius_arr) " \
                                "to find the Outer Seeds from the Prob. Atlas (Certain Region Border of the Background).\n" \
                                 "Ex: -d 0,5,5 or -d 10\nDefault: 0"},
        {.short_name = "-c", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=false, .help="Parameter C of linear SVM. Default 1e-5"},
        {.short_name = "", .long_name = "--t0", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation/Registration File for Initial Transform.\n"\
                                "For multiple parameter files (up to 3), use the sequence of options: --t0, --t1, and --t2"},
        {.short_name = "", .long_name = "--t1", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation/Registration File."},
        {.short_name = "", .long_name = "--t2", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Elastix Transformation/Registration File."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, iftFileSet **atlas_set, char **template_path, char **out_adapro_path) {
    const char *atlas_entry = iftGetConstStrValFromDict("--atlas-entry", args);
    *template_path          = iftGetStrValFromDict("--template", args);
    *out_adapro_path        = iftGetStrValFromDict("--output-adapro", args);

    iftValidateRequiredArgs(atlas_entry, *template_path, *out_adapro_path);

    *atlas_set = iftLoadFileSetFromDirOrCSV(atlas_entry, 0, true);
    if (!(*atlas_set)->n)
        iftError("Input Atlas Set is empty!", "iftGetRequiredArgs");

    puts("-----------------------");
    printf("- Atlas Entry: \"%s\"\n", atlas_entry);
    printf("- Template: \"%s\"\n", *template_path);
    printf("- Output Bank: \"%s\"\n", *out_adapro_path);
    puts("-----------------------");
}


void iftValidateRequiredArgs(const char *atlas_entry, const char *template_path,
                             const char *out_adapro_path) {
    // ATLAS ENTRY
    if (iftFileExists(atlas_entry)) {
        if (!iftEndsWith(atlas_entry, ".csv")) {
            iftError("The Label Image Set Entry File \"%s\" is not a CSV file... Try *.csv", "iftValidateRequiredArgs", atlas_entry);
        }
    }
    else if (!iftDirExists(atlas_entry, 0))
        iftError("Invalid Pathname for the Label Image Set Entry: \"%s\"...\nTry a valid Directory or a CSV file",
                 "iftValidateRequiredArgs", atlas_entry);

    // REFERENCE IMAGE
    if (!iftIsImageFile(template_path))
        iftError("Invalid Reference Image: \"%s\"", "iftValidateRequiredArgs", template_path);

    // OUT BANK PATH
    if (!iftEndsWith(out_adapro_path, ".zip"))
        iftError("Invalid Zip File Extension: \"%s\". Try *.zip", "iftValidateRequiredArgs", iftFileExt(out_adapro_path));

    char *parent_dir = iftParentDir(out_adapro_path);
    if (!iftDirExists(parent_dir, 0))
        iftMakeDir(parent_dir);
    free(parent_dir);
}


void iftGetOptionalArgs(  iftDict *args, const char *label_path, char **train_voxels_img_path,
                        iftIntArray **labels, iftFloatArray **e_radius_arr, iftFloatArray **d_radius_arr, double *C,
                        iftFileSet **elastix_files) {
    if (iftDictContainKey("--training-voxels", args, NULL)) {
        *train_voxels_img_path = iftGetStrValFromDict("--training-voxels", args);
        if (!iftIsImageFile(*train_voxels_img_path))
            iftError("Invalid Image with the training voxels: \"%s\"", "iftGetOptionalArgs", train_voxels_img_path);
        if (!iftFileExists(*train_voxels_img_path))
            iftError("Training Voxels %s does not exist", "iftGetOptionalArgs", *train_voxels_img_path);

        printf("- Training Voxel Image: \"%s\"\n", *train_voxels_img_path);
    }
    else *train_voxels_img_path = NULL;

    // TARGET LABELS
    *labels = iftGetLabelArrayFromCmdLine(args, label_path);
    
    printf("- Objects: [%d", (*labels)->val[0]);
    for (int i = 1; i < (*labels)->n; i++)
        printf(", %d", (*labels)->val[i]);
    puts("]");
    
    iftGetRadius(args, *labels, e_radius_arr, d_radius_arr);

    if (iftDictContainKey("-c", args, NULL))
        *C = iftGetDblValFromDict("-c", args);
    else *C = 1e-5;
    printf("- C: %lf\n", *C);
    
    
    // only incremental values for --tx, starting at 0 until 2, are considered (--t0, --t1, --t2)
    iftSList *SL = iftCreateSList();
    
    int j = 0;
    char opt[16];
    sprintf(opt, "--t%d", j);
    while (iftDictContainKey(opt, args, NULL)) {
        iftInsertSListIntoTail(SL, iftGetStrValFromDict(opt, args));
        j++;
        sprintf(opt, "--t%d", j);
    }

    if (SL->n > 0) {
        *elastix_files = iftCreateFileSet(SL->n);
        for (size_t i = 0; i < (*elastix_files)->n; i++) {
            char *path = iftRemoveSListHead(SL);
            (*elastix_files)->files[i] = iftCreateFile(path);
            iftFree(path);
        }
        iftDestroySList(&SL);
        
        printf("- Elastix Registration Files:\n");
        for (size_t i = 0; i < (*elastix_files)->n; i++)
            printf("[%lu] %s\n", i, (*elastix_files)->files[i]->path);
    }
    else *elastix_files = NULL;
    
    
    puts("-----------------------\n");
}


iftIntArray *iftGetLabelArrayFromCmdLine(  iftDict *args, const char *label_path) {
    iftIntArray *labels = NULL;

    if (iftDictContainKey("--obj-labels", args, NULL)) {
        const char *labels_str = iftGetConstStrValFromDict("--obj-labels", args);

        if (!iftRegexMatch(labels_str, "^([0-9]+,?)+$"))
            iftError("Invalid Labels: %s", "iftGetLabelArrayFromCmdLine", labels_str);
        
        iftSList *SL = iftSplitString(labels_str, ",");
        labels       = iftCreateIntArray(SL->n);
        
        iftSNode *snode = SL->head;
        for (size_t o = 0; o < SL->n; o++) {
            labels->val[o] = atoi(snode->elem);
            snode = snode->next;
        }
        iftDestroySList(&SL);
    }
    else {
        iftImage *label_img = iftReadImageByExt(iftAbsPathname(label_path));
        labels              = iftGetObjectLabels(label_img);
        iftDestroyImage(&label_img);
    }

    return labels;
}


iftFloatArray *iftParseRadiusEntry(const char *radius_entry,   iftIntArray *labels) {
    if (!iftRegexMatch(radius_entry, "^([0-9]+(.[0-9]+)?,?)+$"))
        iftError("Invalid Radius Entry: %s", "iftParseRadiusEntry", radius_entry);

    iftSList *SL = iftSplitString(radius_entry, ",");
    long n_entry_elems = SL->n;
    
    if ((n_entry_elems != 1) && (n_entry_elems != labels->n))
        iftError("Number of radius passed is != Number of Objects from the AdaPro: %d != %d\n" \
                 "Pass a single interval or one for each object\nEntry: %s", "iftParseRadiusEntry",
                 n_entry_elems, labels->n, radius_entry);

    iftFloatArray *radius_arr = iftCreateFloatArray(labels->n);

    for (int i = 0; SL->n != 0; i++)
        radius_arr->val[i] = atof(iftRemoveSListHead(SL));
    iftDestroySList(&SL);

    // only one overall radius was passed, then fill all radius with it
    if (n_entry_elems == 1)
        for (int i = 1; i < labels->n; i++)
            radius_arr->val[i] = radius_arr->val[0];

    return radius_arr;
}


void iftGetRadius(  iftDict *args,   iftIntArray *labels, iftFloatArray **e_radius_arr,
                  iftFloatArray **d_radius_arr) {
    if (iftDictContainKey("--erosion-radius_arr", args, NULL))
        *e_radius_arr = iftParseRadiusEntry(iftGetConstStrValFromDict("--erosion-radius_arr", args), labels);
    else *e_radius_arr = iftCreateFloatArray(labels->n); // 0-value is set

    if (iftDictContainKey("--dilation-radius_arr", args, NULL))
        *d_radius_arr = iftParseRadiusEntry(iftGetConstStrValFromDict("--dilation-radius_arr", args), labels);
    else *d_radius_arr = iftCreateFloatArray(labels->n); // 0-value is set

    printf("- Erosion Radius: %s\n", iftFloatArrayAsString(*e_radius_arr));
    printf("- Dilation Radius: %s\n", iftFloatArrayAsString(*d_radius_arr));
}
/*************************************************************/



















