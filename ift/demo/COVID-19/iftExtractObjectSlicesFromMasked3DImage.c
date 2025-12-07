/*
 * @author Azael Sousa
 * @date July 5, 2021
 */

#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgumentsFromCmdLine(int argc, const char *argv[]);
void iftValidateInputs(const char *vol_img_path, const char *plane_name, const char *out_path);
iftImagePlaneOrientation iftGetImagePlaneFromName(const char *plane_name);
void iftGetArguments(char **vol_img_path, iftImagePlaneOrientation *plane_orientation,
                     int *slice, char **out_path_vol);
/*************************************************************/

int main(int argc, const char *argv[]) {
    iftDict *args              = iftGetArgumentsFromCmdLine(argc, argv);
    const char *vol_img_path   = iftGetConstStrValFromDict("--image", args);
    const char *label_img_path = iftGetConstStrValFromDict("--input-mask",args);
    const char *plane_name     = iftGetConstStrValFromDict("--plane-orientation", args);
    const char *out_path_vol   = iftGetConstStrValFromDict("--output-dir", args);
    const char *out_path_label = iftGetConstStrValFromDict("--output-label", args);
      double th            = iftGetDblValFromDict("--threshold", args);
    char *basename = iftBasename(vol_img_path);
    iftValidateInputs(vol_img_path, label_img_path, plane_name);
    iftMakeDir(out_path_vol);
    iftMakeDir(out_path_label);

    puts("-----------------------");
    printf("- 3D Image: \"%s\"\n", vol_img_path);
    printf("- Label Image: \"%s\"\n", label_img_path);
    printf("- Image Plane Orientation: \"%s\"\n", plane_name);
    printf("- Output image dir: \"%s\"\n", out_path_vol);
    printf("- Output label dir: \"%s\"\n", out_path_label);
    puts("-----------------------");


    iftImagePlaneOrientation plane_orientation = iftGetImagePlaneFromName(plane_name);
    iftImage *vol_img                          = iftReadImageByExt(vol_img_path);
    iftImage *label_img                        = iftReadImageByExt(label_img_path);

    int size;
    if (plane_orientation == IFT_AXIAL)
        size = (int) (vol_img->zsize + 1) / 2;
    else if (plane_orientation == IFT_CORONAL)
        size = (int) (vol_img->ysize + 1) / 2;
    else // sagittal
        size = (int) (vol_img->xsize + 1) / 2;

    for (int s = 0 ; s < size; s++){
	char filename[200];
	iftImage *slice_vol = iftExtractSlice(vol_img,plane_orientation, s);   
	iftImage *slice_label = iftExtractSlice(label_img,plane_orientation, s);
	if (iftMaximumValue(slice_label) == 0){
	    iftDestroyImage(&slice_vol);
	    iftDestroyImage(&slice_label);
	    continue;
	}
	iftIntArray *arr = iftCountLabelSpels(slice_label);
	if (((double)arr->val[1])/arr->val[0] < th){
	    iftDestroyIntArray(&arr);
	    continue;
	}
	iftImage *out       = NULL;
        if (iftDictContainKey("--normalize-image", args, NULL)) {
            int min_val = iftMinimumValue(slice_vol);
            int max_val = iftMaximumValue(slice_vol);
            printf("- Normalizing Slice Image (Linear Stretch) to 8 bits: [%d, %d] to [0, 255]\n", min_val, max_val);
            out = iftLinearStretch(slice_vol, min_val, max_val, 0, 255);
            iftDestroyImage(&slice_vol);
        }
        else out = slice_vol;

	sprintf(filename,"%s/%s_%03d.png",out_path_vol,basename,s);
	iftWriteImageByExt(out, filename);
	iftDestroyImage(&out);

	sprintf(filename,"%s/%s_%03d.png",out_path_label,basename,s);
	iftWriteImageByExt(slice_label,filename);
	iftDestroyImage(&slice_label);
    }
    
    puts("Done...");

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&vol_img);
    iftDestroyImage(&label_img);

    return 0;
}

/************************** SOURCES **************************/
iftDict *iftGetArgumentsFromCmdLine(int argc, const char *argv[]) {
    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="3D Image pathname."},
        {.short_name = "-p", .long_name = "--plane-orientation", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Image Plane: [AXIAL, CORONAL, SAGITTAL]"},
	{.short_name = "-l", .long_name = "--input-mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input mask"},
        {.short_name = "-o", .long_name = "--output-dir", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output dir."},
	{.short_name = "-m", .long_name = "--output-label", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output label dir."},
    {.short_name = "-t", .long_name = "--threshold", .has_arg=true, .arg_type=IFT_DBL_TYPE,
         .required=true, .help="Mask volume threshold [0,1]. Only slices with more than <threshold> mask will be considered."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);
    

    char program_description[2048] = \
        "This program extracts a slice of a given Plane (AXIAL, CORONAL, SAGITTAL) from a 3D Image.\n"
        "The slice is saved as a 2D Image";
    
    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftValidateInputs(const char *vol_img_path, const char *label_img_path, const char *plane_name) {
    if (!iftIsImageFile(vol_img_path))
        iftError("Invalid Image Extension of \"%s\"\nTry *.scn, *.pgm, *.ppm, *.png", "iftValidateInputs", vol_img_path);

    if (!iftIsImageFile(label_img_path))
	iftError("Invalid Image File \"%s\"\n.Check if image exists or if extension is valid.","iftValidateInputs",label_img_path);    

    char *aux = NULL;
    if (!iftRegexMatch(aux = iftUpperString(plane_name), "^(AXIAL|CORONAL|SAGITTAL)$"))
        iftError("Invalid Plane \"%s\"... Try AXIAL, CORONAL, SAGITTAL", "iftValidateInputs",
                 plane_name);
    free(aux);
}


iftImagePlaneOrientation iftGetImagePlaneFromName(const char *plane_name) {
    iftImagePlaneOrientation plane_orientation = IFT_SAGITTAL;

    char *plane_name_upper = iftUpperString(plane_name);
    if (iftRegexMatch(plane_name_upper, "^(AXIAL|CORONAL|SAGITTAL)$")) {
        if (iftCompareStrings(plane_name_upper, "AXIAL")) {
            plane_orientation = IFT_AXIAL;
        }
        else if (iftCompareStrings(plane_name_upper, "CORONAL"))
            plane_orientation = IFT_CORONAL;
        else // SAGITTAL
            plane_orientation = IFT_SAGITTAL;
    }
    else
        iftError("Invalid Plane \"%s\"... Try AXIAL, CORONAL, SAGITTAL", "iftValidateInputs",
                 plane_name);
    free(plane_name_upper);

    return plane_orientation;
}
/*************************************************************/
