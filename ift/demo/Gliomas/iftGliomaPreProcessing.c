#include "ift.h"

#define EXT3D ".nii.gz"

#define PRE_SUBFOLDERS_ENVELOPE     "brain_segm"
#define PRE_SUBFOLDERS_STD_IMGS     "std_imgs"
#define PRE_SUBFOLDERS_PRE_IMGS     "pre_imgs"
#define PRE_SUBFOLDERS_PRE_N4_IMGS     "pre_n4_imgs"
#define PRE_SUBFOLDERS_MSP          "msp"
#define PRE_SUBFOLDERS_DEFPROTOCOL  "def2protocol"
#define PRE_SUBFOLDERS_DEFTEMPLATE  "def2template"

typedef struct ret_gbm {
    iftImage **std_imgs;    //Standardized image (Final)
    iftImage **pre_imgs;   //image before standardized
    iftImage **orig_imgs;   
    iftImage **mask_imgs;
    iftImage **pre_n4_imgs; //preprocessing image but with N4 (bad for segmentation)
    iftImage **reg_imgs;

    iftFileSet **def2t1ce;
    iftFileSet *def2template;
    iftPlane **msp;


} retGBM;

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **input_csv_path, char **adapro_path, char **output_dir_path);
void iftValidateRequiredArgs(const char *input_dir_path, const char *adapro_path, const char *out_dir_path);

iftImage * standardizeBrain(iftImage* in_img);
void iftSetNewRadius(iftAdaPro *adapro);
retGBM* iftGBMPreProcessingArray(char ** path_array, int nimgs, char *adapro_path);
void destroyRetGBM(retGBM *preproc);
void writeDefFields(  iftFileSet *def_fields, char * filename);
void saveFiles(retGBM *preproc, char *out_dir_path, char ** path_array, int nimgs);
retGBM* allocateRetGBM(  int nimgs);
/************************************************************/

int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);
    // required args
    char *input_csv_path = NULL;
    char *adapro_path = NULL;
    char *out_dir_path = NULL;


    iftGetRequiredArgs(args, &input_csv_path, &adapro_path, &out_dir_path);

    iftCSV* csv = iftReadCSV(input_csv_path, ',');
    
    for (int row_count=1;row_count<csv->nrows;row_count++){
        
        retGBM *preproc= iftGBMPreProcessingArray(csv->data[row_count], csv->ncols, adapro_path);
        saveFiles(preproc, out_dir_path, csv->data[row_count], csv->ncols);
        destroyRetGBM(preproc);
    }

    iftDestroyCSV(&csv);
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[IFT_STR_DEFAULT_SIZE] = \
        "iftGBMPreProcessing";

    //TODO: change output to segment 
    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-csv", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Input csv file with images relationship. Input images should be in native space. " \
                                "For a example of csv file check the demo folder."},
        {.short_name = "-b", .long_name = "--adapro", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="AdaPro model used for segmentation (*.zip)."},
        {.short_name = "-o", .long_name = "--output-folder", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname for the output folder, which will save the brain envelope, MSP plane" \
                                "deform. fields and preprocessed files."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **input_csv_path, char **adapro_path, char **output_dir_path) {
    *input_csv_path = iftGetStrValFromDict("--input-csv", args);
    *adapro_path = iftGetStrValFromDict("--adapro", args);
    *output_dir_path = iftGetStrValFromDict("--output-folder", args);
    

    iftValidateRequiredArgs(*input_csv_path, *adapro_path, *output_dir_path);

    puts("-----------------------");
    printf("- Input CSV File: \"%s\"\n", *input_csv_path);
    printf("- AdaPro: \"%s\"\n", *adapro_path);
    printf("- Output Segmented Folder: \"%s\"\n", *output_dir_path);
    puts("-----------------------");
}


void verify_and_create_subdir(const char *parent_dir, const char *subdir){
    //VERIFY INNER FOLDERS - brain envelope
    char basename[200];
    sprintf(basename,"%s/%s",parent_dir,subdir);
    if (!iftDirExists(basename)) {
        printf("Creating the output dir :  %s \n", basename);
        iftMakeDir(basename);
    }
}

void iftValidateRequiredArgs(const char *input_dir_path, const char *adapro_path, const char *out_dir_path) {

    if (!iftFileExists(input_dir_path)) {
        iftError("Invalid Input Image's Pathname: \"%s\"", "iftValidateRequiredArgs", input_dir_path);
    }

    // MODEL
    if (iftFileExists(adapro_path)) {
        if (!iftEndsWith(adapro_path, ".zip")) {
            iftError("Invalid Extension for the AdaPro: \"%s\"... Try *.zip",
                     "iftValidateRequiredArgs", adapro_path);
        }
    }
    else iftError("AdaPro \"%s\" does not exist", "iftValidateRequiredArgs", adapro_path);

    // OUTPUT FOLDER
    if (!iftDirExists(out_dir_path)) {
        printf("Creating the output dir :  %s \n", out_dir_path);
        iftMakeDir(out_dir_path);

        char *subp[7] = {PRE_SUBFOLDERS_ENVELOPE, PRE_SUBFOLDERS_STD_IMGS, PRE_SUBFOLDERS_MSP,
                       PRE_SUBFOLDERS_DEFPROTOCOL, PRE_SUBFOLDERS_DEFTEMPLATE, PRE_SUBFOLDERS_PRE_N4_IMGS, PRE_SUBFOLDERS_PRE_IMGS};

        for (int i =0; i < 7; i++){
            verify_and_create_subdir(out_dir_path, subp[i]);
        }

    }
}
iftImage * standardizeBrain(iftImage* in_img){

    iftImage *input = iftNormalizeWithNoOutliersInRegion(in_img,NULL,0,4095,99.0);

    iftHist  *hist  = iftCalcGrayImageHist(input, NULL, 4096, 4095, false);

    int moda = 1;
    for (int i=moda+1; i < hist->nbins-1; i++)
      if (hist->val[i]>hist->val[moda])
	    moda = i;
    
    iftDestroyHist(&hist);
    
    iftImage *output = iftWindowAndLevel(input,3000,moda,4095);
    //destroyers
    iftDestroyImage(&input);
    
    return output;
}


void iftSetNewRadius(iftAdaPro *adapro) {

    printf("- Erosion Radius: ");
    for (int o = 0; o < adapro->labels->n; o++)
        printf("%f ", adapro->obj_models[o]->erosion_radius);

    printf("\n- Dilation Radius: ");
    for (int o = 0; o < adapro->labels->n; o++)
        printf("%f ", adapro->obj_models[o]->dilation_radius);
    puts("");
    puts("-----------------------");
}

retGBM* allocateRetGBM(  int nimgs){
    //alocate memory
    retGBM *ret = (retGBM*)iftAlloc(1, sizeof(retGBM));

    ret->std_imgs  = (iftImage**) iftAlloc(nimgs, sizeof(iftImage*));
    ret->pre_imgs  = (iftImage**) iftAlloc(nimgs, sizeof(iftImage*));
    ret->orig_imgs = (iftImage**) iftAlloc(nimgs, sizeof(iftImage*));
    ret->mask_imgs = (iftImage**) iftAlloc(nimgs, sizeof(iftImage*));
    ret->pre_n4_imgs  = (iftImage**) iftAlloc(nimgs, sizeof(iftImage*));
    ret->reg_imgs  = (iftImage**) iftAlloc(nimgs, sizeof(iftImage*));

    ret->def2t1ce    = (iftFileSet**) iftAlloc(nimgs-1, sizeof(iftFileSet*));
    ret->def2template = (iftFileSet*) iftAlloc(1, sizeof(iftFileSet*));
    ret->msp           = (iftPlane**) iftAlloc(nimgs, sizeof(iftPlane*));

    return ret;
}

retGBM* iftGBMPreProcessingArray(char ** path_array, int nimgs, char *adapro_path){

    //alocate memory
    retGBM *ret = allocateRetGBM(nimgs);
    
    iftAdaPro *adapro = iftReadAdaPro(adapro_path);
    iftSetNewRadius(adapro);
    //return ret;
    char *tmp_basename = iftMakeTempPathname("ElastixParams.", NULL, NULL);
    printf("baseline adapro %s\n",tmp_basename);
    iftFileSet *elastix_files = iftWriteAdaProElastixParamFiles(adapro, tmp_basename);
    printf("n files %ld \n", elastix_files->n);
    iftFileSet *rigid_files = iftCreateFileSet(1);
    rigid_files->files[0] = iftCopyFile(elastix_files->files[0]);


    int nbits = iftImageDepth(adapro->template_img);
    bool skip_n4 = false; //TODO: alterar e colocar uma MACRO
    bool skip_median_filter = false; //TODO: alterar isso
    bool skip_msp_alignment = false;
    bool skip_hist_matching = true;

    for(int i=0;i<nimgs;i++){
        printf("dealing with image %s\n", path_array[i]);
        ret->orig_imgs[i]  = iftReadImageByExt(path_array[i]);

        //N4+Median filter+MSP
        ret->pre_n4_imgs[i] = iftBrainMRIPreProcessing(ret->orig_imgs[i], nbits, NULL, NULL, NULL, NULL, 
                                                  skip_n4, skip_median_filter, skip_msp_alignment, skip_hist_matching, &ret->msp[i]);
    
        if (i>0){
            //register to T1GD and transform to template
            iftImage *reg_img = iftRegisterImageByElastix(ret->pre_n4_imgs[i], ret->pre_n4_imgs[0], NULL, NULL,
                                                            rigid_files, NULL, &ret->def2t1ce[i-1], NULL);
            iftSed(ret->def2t1ce[i-1]->files[0]->path, "(FinalBSplineInterpolationOrder 0)", "(FinalBSplineInterpolationOrder 3)");
            ret->reg_imgs[i] = iftTransformImageByTransformix(reg_img, ret->def2template->files[0]->path);

            ret->mask_imgs[i] = iftSegmentByAdaPro(ret->reg_imgs[i], adapro, NULL);
            iftDestroyImage(&reg_img);
        }else{
            // First protocol should be register to template directly
            ret->reg_imgs[i] = iftRegisterImageByElastix(ret->pre_n4_imgs[i], adapro->template_img, NULL, NULL,
                                                            rigid_files, NULL, &ret->def2template, NULL);
            iftSed(ret->def2template->files[0]->path, "(FinalBSplineInterpolationOrder 0)", "(FinalBSplineInterpolationOrder 3)");

            ret->mask_imgs[i] = iftSegmentByAdaPro(ret->reg_imgs[i], adapro, NULL);
        }

        //***********************************************
        //Now process original image (without N4 effect)
        //***********************************************

        //Align MSP
        iftImage* tmp_img1 = NULL, *tmp_img2 = NULL;
        tmp_img1 = iftRotateImageToMSP(ret->orig_imgs[i], ret->msp[i], IFT_LINEAR_INTERPOLATION);

        //Transform images to template space
        if (i==0){
            //first protocol - directly transform
            tmp_img2 = iftTransformImageByTransformix(tmp_img1, ret->def2template->files[0]->path);

        }else{
            //other protocol must be transformed to the first protocol (e.g. T1GD) and then transformed to template space
            iftImage *tmp_img3 = iftTransformImageByTransformix(tmp_img1, ret->def2t1ce[i-1]->files[0]->path);

            tmp_img2 = iftTransformImageByTransformix(tmp_img3, ret->def2template->files[0]->path);
            iftDestroyImage(&tmp_img3);
        }
        iftDestroyImage(&tmp_img1);
       
        //Deal with negative values
        int img_min_val = -iftMinimumValue(tmp_img2);
        tmp_img1 = iftAddValue(tmp_img2,img_min_val);
        iftDestroyImage(&tmp_img2);

        //Masking with brain envelope
        tmp_img2 = iftMask(tmp_img1, ret->mask_imgs[i]);
        iftDestroyImage(&tmp_img1);

        //Standardize Images
        tmp_img1 = standardizeBrain(tmp_img2);
        ret->std_imgs[i] = iftCopyImage(tmp_img1);
        ret->pre_imgs[i] = iftCopyImage(tmp_img2);

        iftDestroyImage(&tmp_img2);
        iftDestroyImage(&tmp_img1);
    }


    //cleaning up
    iftFree(tmp_basename);
    iftDestroyAdaPro(&adapro);
    iftRemoveFileSet(elastix_files);
    iftDestroyFileSet(&elastix_files);
    iftDestroyFileSet(&rigid_files);

    return ret;
}



void destroyRetGBM(retGBM *preproc){
    free(preproc->msp);
    free(preproc->def2t1ce);
    free(preproc->def2template);

    free(preproc->pre_n4_imgs);
    free(preproc->mask_imgs);
    free(preproc->orig_imgs);
    free(preproc->std_imgs);
    free(preproc->pre_imgs);
    free(preproc);
}

void writeDefFields(  iftFileSet *def_fields, char * filename){

    iftFileSet *new_def_fields = iftCreateFileSet(def_fields->n);
    for (long i = 0; i < def_fields->n; i++) {
        char out_def_field[512];
        sprintf(out_def_field, "%s.%ld.txt", filename, i);
        new_def_fields->files[i] = iftCreateFile(out_def_field);
        
        printf("[%ld] %s --> %s\n", i, def_fields->files[i]->path, new_def_fields->files[i]->path);
        iftCopyFromDisk(def_fields->files[i]->path, new_def_fields->files[i]->path);
        iftRemoveFile(def_fields->files[i]->path);
        
        if (i > 0)
            iftSed(new_def_fields->files[i]->path, def_fields->files[i - 1]->path, new_def_fields->files[i - 1]->path);
    }
    puts("");
        
    iftDestroyFileSet(&new_def_fields);

}

void saveFiles(retGBM *preproc, char *out_dir_path, char ** path_array, int nimgs){

    char basename[200];
    char strSeparator[2] = {'/', '\0'};
    char strDotSeparator[2] = {'.', '\0'};

    // for image
    for(int i=0;i<nimgs;i++){

        
        iftSList *SL = iftSplitString(path_array[i], strSeparator); 
        char * filename = iftRemoveSListTail(SL);

        //save std file
        sprintf(basename,"%s/%s/%s",out_dir_path,PRE_SUBFOLDERS_STD_IMGS,filename);
        iftWriteImageByExt(preproc->std_imgs[i], basename);

        //save preproc
        sprintf(basename,"%s/%s/%s",out_dir_path,PRE_SUBFOLDERS_PRE_IMGS,filename);
        iftWriteImageByExt(preproc->pre_imgs[i], basename);

        //save mask
        sprintf(basename,"%s/%s/%s",out_dir_path,PRE_SUBFOLDERS_ENVELOPE,filename);
        iftWriteImageByExt(preproc->mask_imgs[i], basename);

        //save preproc with n4
        sprintf(basename,"%s/%s/%s",out_dir_path,PRE_SUBFOLDERS_PRE_N4_IMGS,filename);
        iftWriteImageByExt(preproc->pre_n4_imgs[i], basename);


        iftSList *SL2 = iftSplitString(filename, strDotSeparator); 
        char * filename2 = iftRemoveSListHead(SL2);

        //save msp
        sprintf(basename,"%s/%s/%s%s",out_dir_path,PRE_SUBFOLDERS_MSP,filename2,".json");
        iftWritePlane(preproc->msp[i], basename);

        //save def field
        if (i==0){
            //def2template
            sprintf(basename,"%s/%s/%s",out_dir_path,PRE_SUBFOLDERS_DEFTEMPLATE,filename2);
            writeDefFields(preproc->def2template,basename);
        }else{
            //def2t1ce
            sprintf(basename,"%s/%s/%s",out_dir_path,PRE_SUBFOLDERS_DEFPROTOCOL,filename2);
            writeDefFields(preproc->def2t1ce[i-1], basename);
        }


        iftDestroySList(&SL);
        iftDestroySList(&SL2);
        iftFree(filename);
        iftFree(filename2);
    }

}