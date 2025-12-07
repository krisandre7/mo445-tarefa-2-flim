/*
Created by azael on 18/11/16.

   This software resulted in the paper: ALTIS: A fast and automatic lung and trachea CT‐image segmentation method,
published at the Medical Physics Journal, 2019.

Other related programs:
  iftPleuralAnomalyDetectionByMathMorphology.c
  iftCheckALTISIntegrity.c
*/

#include "ift.h"

#define STANDARD_VOXEL_SIZE         1.25
#define ADJ_REL_RADIUS              1.732    //1.732
#define CLOSING_RADIUS              2.0      //3
#define OTSU_THRESHOLD              1.2      //1.4
#define TRACHEA_THRESHOLD           0.5      //0.7
#define DILATION_SQUARED_RADIUS     256.0    //4.0
#define EROSION_SQUARED_RADIUS      81       //(default 100)
#define N_PREDECESSOR               25

/* --- Not working yet ---
 * The DICOM definition makes the software able to read
 * dicom images for a given directory and unable to further
 * reading scene(.scn) images. Although, the software will
 * take more time to segment the thorax image, this will
 * provide a wider range of image extensions, making it
 * easier to use.
 */
//#define DICOM

/*
 * Uncomment the DEBUG definition to create images step-by-step of
 * the algorithm. Also, a new requeired parameter will be necessary: 
 * the name of the folder holding the dataset. Inside this folder, 
 * there must be another folder called "controls".
 * Inside the folder "controls", the user must create the following folders:
 * -- DEBUG/
 * ------ closed_resp_bin/
 * ------ edt/
 * ------ edt_th/
 * ------ geo_th/
 * ------ int_geo/
 * ------ seeds/
 * ------ trachea/
 * ------ grad/
 * ------ voi/
 * -- <Input image folder>
 * -- <Output folder>
 * All images must be within the <Input image folder>.
 *
 * If the images to be segment are not "control" but patients,
 * change the definition PARENT_DEBUG_FOLDER from "control to
 * "patients" (the patient images must be inside a "patients"
 * folder).
 */
#define DEBUG

/*
 * The PARAMETER_OPTIMIZATION definition is used to optmize all
 * inner parameters within this software. These optimal parameters
 * will be found throughout a series of tests involving a set of
 * training images.
 */
#define PARAMETER_OPTIMIZATION

/*
 * This parameter prints on the terminal every step performed by
 * the software during the lungs and trachea segmentation
 */
#define SHOW_STEP_BY_STEP

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **img_out);
void iftGetOptionalArgs(  iftDict *args);
void iftGaussianFilter(  iftDict *args, iftImage **img);
void iftRemoveNoise(  iftDict *args, iftImage **img, iftAdjRel *A);
void iftCheckDataSet(  iftDict *args);
void iftValidateRequiredArgs(const char *img_path);

void iftCreateDirectory(const char *folder);

iftImage *iftExtractVolumeOfInterest(iftImage *orig, iftAdjRel *A, iftImage **grad, iftSet **S);

iftImage *iftSegmentRespiratorySystem(iftImage *grad, iftImage *voi, iftSet *S, iftAdjRel *A, iftImage *orig, bool improve);

iftLabeledSet *iftLabelMarkers(iftImage *bin, iftSet *Si, iftSet *St, iftSet *Se, iftSet **forbidden);
iftSet *iftTracheaMarker(iftImage *bin, iftSet *Si);
iftImage *iftExternalInternalLungMarkers(iftImage *bin, iftSet *S, iftSet **Si, iftSet **Se);

int iftScanAnomaliesOnLungSegmentation(iftImage *segmentation, iftImage *scn);
/*************************************************************/

/********************* GLOBAL VARIABLES **********************/
#ifdef DEBUG
    char filename[200];
    char base[200];
    char file[200];
    char img_parent_folder[200];
    char img_parent_folder[200];
    char debug_folders[220];
#endif

long total_time;
char *time_path = NULL;
double closing_radius=CLOSING_RADIUS;
double adj_rel_radius=ADJ_REL_RADIUS;
double thresholdOtsu=OTSU_THRESHOLD;
double thresholdTrachea=TRACHEA_THRESHOLD;
long dilationSquaredRadius=DILATION_SQUARED_RADIUS;
long erosionSquaredRadius=EROSION_SQUARED_RADIUS;
long n_predecessors=N_PREDECESSOR;
bool improve = false;
/**************************************************************/

int main (int argc, const char **argv)
{
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path          = NULL;
    char *img_out           = NULL;
    char *formatted         = NULL;
    iftImage *scn = NULL, *grad = NULL, *voi = NULL, *segm = NULL;
    iftAdjRel *A = NULL;
    timer *t1,*t2;

    // Variables Initialization

    iftGetRequiredArgs(args,&img_path,&img_out);

    iftGetOptionalArgs(args);

    #ifdef DEBUG
        char *pointer = strrchr(img_path,'/');
        if (pointer != NULL){
            char aux[100];
            strcpy(file,pointer+1);
            strncpy(aux,img_path,(pointer-img_path));
            pointer = strrchr(aux,'/');
            if (pointer != NULL){
                strncpy(img_parent_folder,aux,(pointer-aux));
            } else {
                strcpy(img_parent_folder,".");
            }
        } else {
            strcpy(img_parent_folder,".");
            strcpy(file,img_path);
        }
        strcpy(base,file);
        pointer = strrchr(base,'.');
        if (pointer != NULL)
            *pointer = '\0';
    #endif

    #ifdef DEBUG
        puts("- Creating DEBUG folders.");
        sprintf(debug_folders,"%s/DEBUG/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        sprintf(debug_folders,"%s/DEBUG/dilate/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        sprintf(debug_folders,"%s/DEBUG/externalInternalMarkers/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        sprintf(debug_folders,"%s/DEBUG/trachea/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        sprintf(debug_folders,"%s/DEBUG/markers/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        sprintf(debug_folders,"%s/DEBUG/grad/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        sprintf(debug_folders,"%s/DEBUG/voi/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        sprintf(debug_folders,"%s/DEBUG/geodesic/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        sprintf(debug_folders,"%s/DEBUG/waterlabel/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        sprintf(debug_folders,"%s/DEBUG/new_markers/",img_parent_folder);
        printf("--- %s\n",debug_folders);
        iftCreateDirectory(debug_folders);
        printf("\n");
    #endif

    scn = iftReadImageByExt(img_path);
    iftFree(img_path);

    // Checking if input image is isotropic
    if (!iftIsIsotropic(scn)){
        puts("\nWarning in iftALTIS");
        puts("   Input Image is must be Isotropic (dx = dy = dz).");
        printf("   Voxel size of input image: (%.2f, %.2f, %.2f)\n",scn->dx,scn->dy,scn->dz);
        puts("   The result may not be as expected.");
    }

    if (iftDictContainKey("--execution-time", args, NULL)) {
        time_path = iftGetStrValFromDict("--execution-time",args);
        total_time = 0;
    }

    if (iftDictContainKey("--improve-segmentation", args, NULL)){
        improve = true;
        n_predecessors = iftGetLongValFromDict("--improve-segmentation", args);
    }

    // Creating Adjacency Realtion
    A = iftSpheric((adj_rel_radius*scn->dx/STANDARD_VOXEL_SIZE)<1?1:(adj_rel_radius*scn->dx/STANDARD_VOXEL_SIZE));

    // Image Pre-processing
    iftGaussianFilter(args, &scn);
    iftRemoveNoise(args, &scn, A);

    /* --------------------------------------------   Start ALTIS   -------------------------------------------- */
    t1 = iftTic();

    // Respiratory System Extraction
    iftSet *S = NULL;
    #ifdef SHOW_STEP_BY_STEP
        puts("");
        puts("Extracting Volume of Interest");
    #endif
    voi   = iftExtractVolumeOfInterest(scn,A,&grad,&S);

    #ifdef DEBUG
        sprintf(filename,"%s/DEBUG/voi/%s",img_parent_folder,file);
        iftWriteImageByExt(voi,filename);
        sprintf(filename,"%s/DEBUG/grad/%s",img_parent_folder,file);
        iftWriteImageByExt(grad,filename);
    #endif
    
    // Lungs and Trachea Segmentation
    #ifdef SHOW_STEP_BY_STEP
        puts("");
        puts("Segmenting Respiratory System");
    #endif
    segm = iftSegmentRespiratorySystem(grad, voi, S, A, scn, improve);
    
    t2 = iftToc();
    /* --------------------------------------------   End ALTIS   -------------------------------------------- */

    //Exporting Time
    if (time_path != NULL){
        #ifdef SHOW_STEP_BY_STEP
            puts("");
            puts("Exporting Time File");
        #endif
        puts("");
        puts("- Exporting Segmentation Execution Time");
        total_time = (t2->tv_sec-t1->tv_sec)*1000.0 +
                     (t2->tv_usec-t1->tv_usec)*0.001;
        FILE *fp = fopen(time_path, "a+");
        fprintf(fp, "%ld\n", total_time);
        fclose(fp);
    }

    // Show computational time
    puts("\nDone...");
    printf("Total Thorax Segmentation took ");
    formatted = iftFormattedTime(iftCompTime(t1, t2));
    puts(formatted);
    iftFree(formatted);

    #ifdef SHOW_STEP_BY_STEP
        puts("");
        puts("Exporting Segmentation");
    #endif
    // Saving Segmentation Result
    iftCopyVoxelSize(scn,segm);
    iftWriteImageByExt(segm,img_out);
    iftFree(img_out);

    #ifdef SHOW_STEP_BY_STEP
        puts("");
        puts("Deallocating Memory Space");
        puts("");
    #endif
    // Deallocation Memory
    iftDestroyImage(&voi);
    iftDestroyImage(&grad);
    iftDestroyImage(&segm);
    iftDestroyImage(&scn);
    iftDestroyAdjRel(&A);

    return 0;
}

//---------------------------------------------------------------------------
//                    Program Arguments and Parameters
//---------------------------------------------------------------------------

void iftCreateDirectory(const char *folder)
{
    mkdir(folder,0777);
}

iftDict *iftGetArgs(int argc, const char *argv[])
{
    char program_description[2048] = \
        "- This program segments the respiratory system applying different labels for the right lung, left lung and trachea.\n" \
        "- You can pass the scene (.scn) file containing a chest CT scans.\n";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input Scene Image."},
            {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output Scene Image."},
            {.short_name = "-g", .long_name = "--gaussian-filter", .has_arg=false, .arg_type=IFT_UNTYPED,
                    .required=false, .help="Preprocess the input image with a gaussian filter."},
            {.short_name = "-n", .long_name = "--noise-filter", .has_arg=false, .arg_type=IFT_UNTYPED,
                    .required=false, .help="Value to reduce noise of the input image."},
	        #ifdef PARAMETER_OPTIMIZATION
                {.short_name = "-a", .long_name = "--adj-rel-radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                        .required=false, .help="Adjacency relationship of pixels.\nDefault: 1.73."},
                {.short_name = "-c", .long_name = "--closing-radius", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                        .required=false, .help="Radius for morphological closure.\nDefault: 2."},
                {.short_name = "-s", .long_name = "--otsu-threshold", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                        .required=false, .help="Otsu threshold value.\nDefault: 1.2."},
                {.short_name = "-t", .long_name = "--trachea-threshold", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                        .required=false, .help="Geodesic distance threshold value for trachea location.\nDefault: 0.7."},
                {.short_name = "-d", .long_name = "--dilation-squared-radius", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                       .required=false, .help="Radius to estimate outer seeds.\nDefault: 9.0 (3^3)."},
                {.short_name = "-e", .long_name = "--erosion-squared-radius", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                       .required=false, .help="Radius to estimate inner seeds.\nDefault: 100 (10^2)."},
            #endif
            {.long_name = "--improve-segmentation",.has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=false, .help="Improve the segmentation by performing a intensity-based IFT. Define the number of predecessors (Default:25)"},
            {.long_name = "--consistency-test",.has_arg=false, .arg_type=IFT_UNTYPED,
                    .required=false, .help="Apply the consistency tests on the segmentation result."},
            {.long_name = "--execution-time", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Saves the total execution time in a file."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **img_out)
{
    *img_path        = iftGetStrValFromDict("--input-img", args);
    *img_out         = iftGetStrValFromDict("--output-img", args);

    iftValidateRequiredArgs(*img_path);

    puts("");
    puts("--------------------------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Output Image: \"%s\"\n", *img_out);
    puts("--------------------------------------");
    puts("");
}

void iftGetOptionalArgs(  iftDict *args)
{
    #ifdef PARAMETER_OPTIMIZATION
        if (iftDictContainKey("--adj-rel-radius", args, NULL))
            adj_rel_radius = iftGetDblValFromDict("--adj-rel-radius", args);
        else
    #endif
        adj_rel_radius = ADJ_REL_RADIUS;

    #ifdef PARAMETER_OPTIMIZATION
        if (iftDictContainKey("--closing-radius", args, NULL))
            closing_radius = iftGetLongValFromDict("--closing-radius", args);
        else
    #endif
        closing_radius = CLOSING_RADIUS;

    #ifdef PARAMETER_OPTIMIZATION
        if (iftDictContainKey("--otsu-threshold", args, NULL))
            thresholdOtsu = iftGetDblValFromDict("--otsu-threshold", args);
        else
    #endif
        thresholdOtsu = OTSU_THRESHOLD;

    #ifdef PARAMETER_OPTIMIZATION
        if (iftDictContainKey("--trachea-threshold", args, NULL))
            thresholdTrachea = iftGetDblValFromDict("--trachea-threshold", args);
        else
    #endif
        thresholdTrachea = TRACHEA_THRESHOLD;

    #ifdef PARAMETER_OPTIMIZATION
        if (iftDictContainKey("--dilation-squared-radius", args, NULL))
            dilationSquaredRadius = iftGetLongValFromDict("--dilation-squared-radius", args);
        else
    #endif
        dilationSquaredRadius = DILATION_SQUARED_RADIUS;

    #ifdef PARAMETER_OPTIMIZATION
        if (iftDictContainKey("--erosion-squared-radius", args, NULL))
            erosionSquaredRadius = iftGetLongValFromDict("--erosion-squared-radius", args);
        else
    #endif
        erosionSquaredRadius = EROSION_SQUARED_RADIUS;


    if (iftDictContainKey("--execution-time", args, NULL)) {
        time_path = iftGetStrValFromDict("--execution-time",args);
        total_time = 0;
    }

    if (iftDictContainKey("--improve-segmentation", args, NULL)){
        improve = true;
        n_predecessors = iftGetLongValFromDict("--improve-segmentation", args);
    }

}

void iftValidateRequiredArgs(const char *img_path)
{
    if (!iftIsImageFile(img_path))
        iftError("Invalid Input Image: \"%s\"", "iftValidateRequiredArgs", img_path);
}

void iftGaussianFilter(  iftDict *args, iftImage **img)
{
    if (iftDictContainKey("--gaussian-filter", args, NULL)) {
        #ifdef SHOW_STEP_BY_STEP
            puts("");
            puts("- Performing the Gaussian Filter");
            timer     *t1=iftTic();
        #endif
        iftImage *gauss, *aux, *scn;
        scn = *img;
        iftKernel *K = iftGaussianKernel(ADJ_REL_RADIUS,1.5);
        gauss = iftLinearFilter(scn,K);
        iftDestroyKernel(&K);
        aux = iftAbs(gauss);
        iftDestroyImage(&gauss);
        *img = aux;
        iftDestroyImage(&scn);
        #ifdef SHOW_STEP_BY_STEP
            printf("--- Gaussian filter took ");
            char *formatted = iftFormattedTime(iftCompTime(t1, iftToc()));
            puts(formatted);
            iftFree(formatted);
        #endif
    }
}

void iftRemoveNoise(  iftDict *args, iftImage **img, iftAdjRel *A)
{
    if (iftDictContainKey("--noise-filter", args, NULL)) {
        #ifdef SHOW_STEP_BY_STEP
            puts("");
            puts("- Removing noise from image");
            timer     *t1=iftTic();
        #endif
        iftImage *filt = iftMedianFilter(*img,A);
        iftDestroyImage(img);
        *img = filt;
        #ifdef SHOW_STEP_BY_STEP
            printf("--- Median filter took ");
            char *formatted = iftFormattedTime(iftCompTime(t1, iftToc()));
            puts(formatted);
            iftFree(formatted);
        #endif
    }
}

//---------------------------------------------------------------------------
//                     Respiratory System Extraction
//---------------------------------------------------------------------------

iftImage *iftSortComponentsByAreaVolume(iftImage *label_map)
{
    iftIntArray *index = iftCreateIntArray(iftMaximumValue(label_map)+1);
    iftIntArray *areavolume = iftCreateIntArray(iftMaximumValue(label_map)+1);

    // initializing
    for (int i = 1; i <= iftMaximumValue(label_map); i++){
        index->val[i] = i;
    }
    for (int i = 0; i < label_map->n; i++){
        if (label_map->val[i] > 0)
            areavolume->val[label_map->val[i]]++;
    }

    for (int i = 1; i < areavolume->n; i++)
        for (int j = i-1; j > 0; j--)
            if (areavolume->val[i] > areavolume->val[j]){
                int aux = index->val[i];
                index->val[i] = index->val[j];
                index->val[j] = aux;
                aux = areavolume->val[i];
                areavolume->val[i] = areavolume->val[j];
                areavolume->val[j] = aux;
            }
    iftDestroyIntArray(&areavolume);

    iftImage *out = iftCopyImage(label_map);
    for (int i = 0; i < out->n; i++)
        out->val[i] = index->val[out->val[i]];
    iftDestroyIntArray(&index);

    return out;
}

iftImage *iftExtractVolumeOfInterest(iftImage *img, iftAdjRel *A, iftImage **grad, iftSet **S)
{
    timer *t1,*t2;
    char *formatted = NULL;
    t1 = iftTic();

    #ifdef SHOW_STEP_BY_STEP
        puts("--- Enhancing Volume of Interest");
    #endif
    /* Enhance volume of interest */
    iftImage *enhanced = iftCreateImage(img->xsize,img->ysize,img->zsize);
    iftCopyVoxelSize(img,enhanced);
    #pragma omp parallel for shared(img)
        for (int z=0; z < img->zsize; z++) {
            iftImage *slice = iftGetXYSlice(img,z);
            iftImage *cbas  = iftCloseBasins(slice,NULL,NULL);
            iftImage *res   = iftSub(cbas,slice);
            iftDestroyImage(&slice);
            iftDestroyImage(&cbas);
            iftPutXYSlice(enhanced,res,z);
            iftDestroyImage(&res);
        }

    #ifdef SHOW_STEP_BY_STEP
        puts("--- Computing Gradient Image");
    #endif
    /* Compute output gradient image */
	iftAdjRel *B = iftSpheric(1.732); // 1.732
	*grad        = iftImageBasins(enhanced,B);
	iftDestroyAdjRel(&B);

    #ifdef SHOW_STEP_BY_STEP
        puts("--- Extracting VOI");
    #endif
    /* Extract volume of interest and its boundary as a set of voxels */
    iftImage *bin[2], *voi;
    bin[0] = iftThreshold(enhanced,thresholdOtsu*iftOtsu(enhanced),IFT_INFINITY_INT,1);
    iftDestroyImage(&enhanced);

    int obj_index = 1;
    bin[1] = iftSelectKLargestComp(bin[0],A,2);
    iftImage *comps = iftLabelComp(bin[1], A);
    iftImage *sorted_comps = iftSortComponentsByAreaVolume(comps);
    iftDestroyImage(&comps);
    if (iftMaximumValue(sorted_comps) > 1){
        float obj1_vol = iftAreaVolumeOfObject(sorted_comps,1);
        float obj2_vol = iftAreaVolumeOfObject(sorted_comps,2);
        if ((obj2_vol/obj1_vol) > 0.3) {
            #ifdef SHOW_STEP_BY_STEP
                puts("----- Stretcher detected.");
            #endif
            iftVoxel gc;
            iftBoundingBox bb = iftMinObjectBoundingBox(sorted_comps,1,&gc);
            if (((bb.begin.x == 0) && (bb.end.x = sorted_comps->xsize-1)) ||
                ((bb.begin.y == 0) && (bb.end.y = sorted_comps->ysize-1)) ||
                ((bb.begin.z == 0) && (bb.end.z = sorted_comps->zsize-1)))
                obj_index = 2;
        }
    }
    iftImage *obj = iftExtractObject(sorted_comps, obj_index);
    iftDestroyImage(&bin[1]);
    bin[1] = iftThreshold(obj, 1, 2, 1);
    iftDestroyImage(&obj);

    iftDestroyImage(&bin[0]);
    *S     = NULL;
    bin[0] = iftDilateBin(bin[1],S,closing_radius);
    iftDestroyImage(&bin[1]);
    bin[1] = iftErodeBin(bin[0],S,closing_radius);
    voi    = iftCloseBasins(bin[1],NULL,NULL);
    iftDestroyImage(&bin[0]);
    iftDestroyImage(&bin[1]);

    t2 = iftToc();
    fprintf(stdout,"Extraction of the Volume of Interest took ");
    formatted = iftFormattedTime(iftCompTime(t1, t2));
    puts(formatted);
    iftFree(formatted);

    return(voi);
}

//---------------------------------------------------------------------------
//                     Lungs and Trachea Segmentation
//---------------------------------------------------------------------------

float iftArcWeightByPathMean(iftImage *mimg,iftImage *pred,int p,int q,iftGQueue *Q, int npredecessors)
{
    float    mean, nelems=1, weight=0.0;

    mean=mimg->val[p];

    p = pred->val[p];
    while ((pred->val[p] != IFT_NIL)&&(nelems <= npredecessors)) {
        mean += mimg->val[p];
        nelems++;
        p = pred->val[p];
    }

    mean=mean/nelems;

    weight = fabs(mimg->val[q]-mean);

    return(weight);
}

iftImage *iftImproveSegmentation(iftImage *img, iftLabeledSet *S, int npredecessors, iftSet *forbidden, iftImage *waterlabel)
{
    iftImage      *cost = NULL, *label = NULL, *pred = NULL;
    iftGQueue     *Q    = NULL;
    int            i, p, q, tmp, Imax = iftMaximumValue(img);
    float          weight;
    iftVoxel       u, v;
    iftLabeledSet *seeds=S;
    iftAdjRel     *A = NULL;

    // Initialization

    if (iftIs3DImage(img)) {
        A = iftSpheric(1.0);
    } else {
        A = iftCircular(1.0);
    }

    cost   = iftCreateImage(img->xsize, img->ysize, img->zsize);
    label  = iftCreateImage(img->xsize, img->ysize, img->zsize);
    pred   = iftCreateImage(img->xsize, img->ysize, img->zsize);
    Q      = iftCreateGQueue(Imax, cost->n, cost->val);
    iftSetImage(cost,IFT_INFINITY_INT);

    while (seeds != NULL)
    {
        p               = seeds->elem;
        label->val[p]   = seeds->label;
        pred->val[p]    = IFT_NIL;	
        cost->val[p]    = 0;
        iftInsertGQueue(&Q, p);
        seeds = seeds->next;
    }

    iftSet *F = forbidden;
    while (F != NULL) {
        p   =  F->elem;
        cost->val[p] = IFT_INFINITY_INT_NEG;
        F = F->next;
    }

    // Image Foresting Transform

    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(cost, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(cost, v))
            {
                q = iftGetVoxelIndex(cost, v);

                if (Q->L.elem[q].color != IFT_BLACK){
                    weight      = abs(img->val[q]-img->val[p]);
                    tmp         = iftMax(cost->val[p],iftRound(weight));
                    if (tmp < cost->val[q])
                    {
		                if (Q->L.elem[q].color == IFT_GRAY)
                        {
                			iftRemoveGQueueElem(Q,q);
		                }
		                label->val[q]    = label->val[p];
		                pred->val[q]     = p;
		                cost->val[q]     = tmp;
		                iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&cost);
    iftDestroyImage(&pred);

    iftCopyVoxelSize(img, label);

    for (i = 0; i < waterlabel->n; i++)
        if (waterlabel->val[i] == 3)
            label->val[i] = 3;

    return(label);
}

iftImage *iftImproveSegmentationByPathMean(iftImage *img, iftLabeledSet *S, int npredecessors, iftSet *forbidden, iftImage *waterlabel, iftImage *pred)
{
    iftImage      *cost = NULL, *label = NULL;//, *pred = NULL;
    iftGQueue     *Q    = NULL;
    int            i, p, q, tmp, Imax = iftMaximumValue(img);
    float          weight;
    iftVoxel       u, v;
    iftLabeledSet *seeds=S;
    iftAdjRel     *A = NULL;

    // Initialization

    if (iftIs3DImage(img)) {
        A = iftSpheric(1.0);
    } else {
        A = iftCircular(1.0);
    }

    cost   = iftCreateImage(img->xsize, img->ysize, img->zsize);
    label  = iftCreateImage(img->xsize, img->ysize, img->zsize);
    //pred   = iftCreateImage(img->xsize, img->ysize, img->zsize);
    Q      = iftCreateGQueue(Imax, cost->n, cost->val);
    iftSetImage(cost,IFT_INFINITY_INT);

    while (seeds != NULL)
    {
        p               = seeds->elem;
        label->val[p]   = seeds->label;
        //pred->val[p]    = IFT_NIL;
        cost->val[p]    = 0;
        iftInsertGQueue(&Q, p);
        seeds = seeds->next;
    }

    iftSet *F = forbidden;
    while (F != NULL) {
        p   =  F->elem;
        cost->val[p] = IFT_INFINITY_INT_NEG;
        F = F->next;
    }

    // Image Foresting Transform

    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(cost, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(cost, v))
            {
                q = iftGetVoxelIndex(cost, v);

                if ((Q->L.elem[q].color != IFT_BLACK) && ((waterlabel->val[q] == waterlabel->val[p]) || (waterlabel->val[q] == 0))){
                    weight      = iftArcWeightByPathMean(img,pred,p,q,Q,npredecessors);
                    tmp         = iftMax(cost->val[p],iftRound(weight));
                    if (tmp < cost->val[q])
                    {
                        if (Q->L.elem[q].color == IFT_GRAY){
                            iftRemoveGQueueElem(Q,q);
                        }
                        label->val[q]    = label->val[p];
                        //pred->val[q]     = p;
                        cost->val[q]     = tmp;
                        iftInsertGQueue(&Q, q);
                    }

                }
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&cost);

    //iftDestroyMImage(&mimg);

    iftCopyVoxelSize(img, label);

    for (i = 0; i < waterlabel->n; i++)
        if (waterlabel->val[i] == 3)
            label->val[i] = 3;

    return(label);
}

iftLabeledSet *iftPrepareSeedsForImprovement(iftImage *img, iftImage *label, iftSet *Se, iftAdjRel *A, iftImage *pred, int n_pred, iftSet **forbidden)
{
    iftLabeledSet *newS = NULL;
    iftSet *Saux        = NULL;
    iftImage *leaves    = iftLeafVoxelsOnMask(pred, A, label);


    iftMaskImageToSet(leaves, &Saux);
    iftDestroyImage(&leaves);
    
    while (Saux != NULL) {
      int p = iftRemoveSet(&Saux);
      int q = p, n = 0;
      while((pred->val[q] != IFT_NIL) && (n < n_pred)){
	      q = pred->val[q];
          n++;
      }
      iftInsertLabeledSet(&newS,q,label->val[q]);
    }

    Saux = Se;
    while(Saux != NULL) {
      int p = Saux->elem;
      iftInsertLabeledSet(&newS,p,label->val[p]);
      Saux = Saux->next;
    }

    return (newS);
}

iftImage *iftSegmentRespiratorySystem(iftImage *grad, iftImage *voi, iftSet *S, iftAdjRel *A, iftImage *img, bool improve) {

    timer *t1,*t2;
    char *formatted = NULL;
    iftImage *label=NULL, *dil=NULL;
    iftImage *pred=NULL;
    iftSet   *Si=NULL, *Se=NULL, *St=NULL, *forbidden=NULL;
    iftLabeledSet *seeds = NULL;
    iftLabeledSet *newS = NULL;

    t1 = iftTic();

    #ifdef SHOW_STEP_BY_STEP
        puts("--- Estimating External and Internal Lung Markers");
    #endif
    dil   = iftExternalInternalLungMarkers(voi, S, &Si, &Se);

    #ifdef DEBUG
        sprintf(filename,"%s/DEBUG/dilate/%s",img_parent_folder,file);
        iftWriteImageByExt(dil,filename);
    #endif

    #ifdef SHOW_STEP_BY_STEP
        puts("--- Estimating Trachea Markers");
    #endif
    St    = iftTracheaMarker(voi,Si);

    #ifdef SHOW_STEP_BY_STEP
        puts("--- Labeling Markers");
    #endif
    seeds = iftLabelMarkers(voi, Si, St, Se, &forbidden);
    
    #ifdef DEBUG
        sprintf(filename,"%s/DEBUG/markers/%s.txt",img_parent_folder,base);
        iftWriteSeeds(seeds,img,filename);
    #endif

    #ifdef SHOW_STEP_BY_STEP
        puts("--- Performing Watershed");
    #endif
    label = iftWatershedWithPredMap(grad, A, seeds, forbidden,&pred);

    
    if (improve)
    {
        iftImage *waterlabel = iftCopyImage(label);
        iftDestroyImage(&label);

        #ifdef DEBUG
            sprintf(filename,"%s/DEBUG/waterlabel/%s.scn",img_parent_folder,base);
            iftWriteImageByExt(waterlabel,filename);
        #endif

        #ifdef SHOW_STEP_BY_STEP
        puts("--- Preparing Seeds for Improvement");
        #endif
        newS = iftPrepareSeedsForImprovement(img,waterlabel, Se, A, pred, n_predecessors, &forbidden);

        #ifdef DEBUG
            sprintf(filename,"%s/DEBUG/new_markers/%s.txt",img_parent_folder,base);
            iftWriteSeeds(seeds,img,filename);
        #endif

        #ifdef SHOW_STEP_BY_STEP
        puts("--- Improving Segmentation with a Seed Competition by Mean Path");
        #endif
        label = iftImproveSegmentation(img,newS,n_predecessors,forbidden,waterlabel);
	iftWriteImageByExt(label,"second_label.scn");
	
        iftDestroyImage(&waterlabel);
        iftDestroyImage(&pred);
        iftDestroyLabeledSet(&newS);
    }

    #ifdef SHOW_STEP_BY_STEP
        puts("--- Deallocating memory space");
    #endif
    iftDestroyImage(&dil);
    iftDestroyLabeledSet(&seeds);
    iftDestroySet(&forbidden);
    iftDestroySet(&Si);
    iftDestroySet(&St);
    iftDestroySet(&Se);

    t2 = iftToc();
    fprintf(stdout,"Respiratory System Segmentation took ");
    formatted = iftFormattedTime(iftCompTime(t1, t2));
    puts(formatted);
    iftFree(formatted);

    return label;
}

iftImage *iftExternalInternalLungMarkers(iftImage *bin, iftSet *S, iftSet **Si, iftSet **Se)
{
    iftImage *dist=NULL,*root=NULL,*dil=NULL;
    iftGQueue *Q=NULL;
    int i,p,q,tmp;
    iftVoxel u,v,r;
    iftSet *Saux;
    iftAdjRel *A;

    *Si   = *Se = NULL;
    dist  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
    dil   = iftCopyImage(bin);
    root  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
    Q     = iftCreateGQueue(IFT_QSIZE,bin->n,dist->val);

    for (p=0; p < dist->n; p++) {
        dist->val[p]= IFT_INFINITY_INT;
    }

    /* Use the boundary of the mask and its neighbors outside the mask
       as seeds */

    Saux = S;
    A     = iftSpheric(1.0);
    while (Saux != NULL) {
        p = Saux->elem;
        dist->val[p]=0;
        root->val[p]=p;
        iftInsertGQueue(&Q,p);
        u = iftGetVoxelCoord(bin,p);
        for (i=1; i < A->n; i++){
            v = iftGetAdjacentVoxel(A,u,i);
            if (iftValidVoxel(bin,v)){
                q = iftGetVoxelIndex(bin,v);
                if ((bin->val[q]==0)&&
                    (Q->L.elem[q].color==IFT_WHITE)){
                    dist->val[q]=0;
                    root->val[q]=q;
                    iftInsertGQueue(&Q,q);
                }
            }
        }
        Saux = Saux->next;
    }
    iftDestroyAdjRel(&A);
    A = iftSpheric(sqrtf(3.0));


    /* Compute the internal and external seeds */

    while(!iftEmptyGQueue(Q)) {

        p=iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(bin,p);
        r = iftGetVoxelCoord(bin,root->val[p]);

        if (bin->val[p]==0){ /* external voxel */
            dil->val[p]=1;
            if (dist->val[p] <= dilationSquaredRadius){
                for (i=1; i < A->n; i++){
                    v = iftGetAdjacentVoxel(A,u,i);
                    if (iftValidVoxel(bin,v)){
                        q = iftGetVoxelIndex(bin,v);
                        if ((dist->val[q] > dist->val[p])&&
                            (bin->val[q]==0)){
                            tmp = iftSquaredVoxelDistance(v,r);
                            if (tmp < dist->val[q]){
                                if (dist->val[q] != IFT_INFINITY_INT)
                                    iftRemoveGQueueElem(Q, q);
                                dist->val[q]  = tmp;
                                root->val[q]  = root->val[p];
                                iftInsertGQueue(&Q, q);
                            }
                        }
                    }
                }
            }
            else {
                iftInsertSet(Se,p);
            }
        } else { /* interval voxel */
            if (dist->val[p] <= erosionSquaredRadius){
                for (i=1; i < A->n; i++){
                    v = iftGetAdjacentVoxel(A,u,i);
                    if (iftValidVoxel(bin,v)){
                        q = iftGetVoxelIndex(bin,v);
                        if ((dist->val[q] > dist->val[p])&&
                            (bin->val[q]!=0)){
                            tmp = iftSquaredVoxelDistance(v,r);
                            if (tmp < dist->val[q]){
                                if (dist->val[q] != IFT_INFINITY_INT)
                                    iftRemoveGQueueElem(Q, q);
                                dist->val[q]  = tmp;
                                root->val[q]  = root->val[p];
                                iftInsertGQueue(&Q, q);
                            }
                        }
                    }
                }
            }
            else {
                iftInsertSet(Si,p);
            }
        }
    }

    #ifdef DEBUG
        iftImage *extint = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
        Saux = *Si;
        while (Saux != NULL){
            p = Saux->elem;
            extint->val[p] = 1;
            Saux = Saux->next;
        }
        Saux = *Se;
        while (Saux != NULL){
            p = Saux->elem;
            extint->val[p] = 2;
            Saux = Saux->next;
        }
        sprintf(filename,"%s/DEBUG/externalInternalMarkers/%s",img_parent_folder,file);
        iftWriteImageByExt(extint,filename);
        iftDestroyImage(&extint);
    #endif

    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&root);
    iftDestroyImage(&dist);

    return(dil);
}

iftSet *iftTracheaMarker(iftImage *bin, iftSet *Si)
{
    iftImage  *dist=NULL;
    iftGQueue *Q=NULL;
    int i,p,q,tmp,z,n;
    iftVoxel u,v;
    iftSet *Saux, *St=NULL;
    iftAdjRel *A;

    dist  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
    Q     = iftCreateGQueue(IFT_QSIZE,bin->n,dist->val);

    /* initialize distance map and compute the z coordinate of the
       geodesic center of the mask */

    for (p=0,z=0,n=0; p < bin->n; p++) {
        if (bin->val[p]!=0){
            u = iftGetVoxelCoord(bin,p);
            z = z + u.z; n = n + 1;
            dist->val[p]= IFT_INFINITY_INT;
        }
    }
    z = z / n;
    Saux = Si;

    while (Saux != NULL) {
        p = Saux->elem;
        dist->val[p]=0;
        iftInsertGQueue(&Q,p);
        Saux = Saux->next;
    }

    A     = iftSpheric(sqrtf(3.0));

    /* Compute the geodesic distance transform from the lung seeds */

    while(!iftEmptyGQueue(Q)) {

        p=iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(bin,p);

        for (i=1; i < A->n; i++){
            v = iftGetAdjacentVoxel(A,u,i);
            if (iftValidVoxel(bin,v)){
                q = iftGetVoxelIndex(bin,v);
                if ((dist->val[q] > dist->val[p])&&
                    (bin->val[q]!=0)){
                    tmp = dist->val[p] + (int)(10*iftVoxelDistance(u,v));
                    if (tmp < dist->val[q]){
                        if (dist->val[q] != IFT_INFINITY_INT)
                            iftRemoveGQueueElem(Q, q);
                        dist->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }
    iftDestroyGQueue(&Q);
    //iftDestroyAdjRel(&A);

    /* Find the maximum distance value above the z coordinate of the
       geometric center of the mask */

    tmp = 0;
    for (p=0; p < bin->n; p++){
        if (bin->val[p]!=0){
            u = iftGetVoxelCoord(bin,p);
            if ((u.z >= z)&&
                (dist->val[p]>tmp))
                tmp = dist->val[p];
        }
    }

    /* Select as trachea marker all voxels whose distance from the lung
       marker is above 70% of the maximum distance and z coordinate is
       above the geometric center */

    tmp = thresholdTrachea*tmp;
    for (p=0; p < bin->n; p++) {
        if ((u.z >= z) && (dist->val[p] > tmp) && (bin->val[p] != 0)) {
            iftInsertSet(&St, p);
        }
    }


    #ifdef DEBUG
        iftImage *export_trachea = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
        Saux = St;
        while (Saux != NULL){
            p = Saux->elem;
            export_trachea->val[p] = 1;
            Saux = Saux->next;
        }
        sprintf(filename,"%s/DEBUG/trachea/%s",img_parent_folder,file);
        iftWriteImageByExt(export_trachea,filename);
        for (p=0; p < dist->n; p++){
            if (dist->val[p] == IFT_INFINITY_INT)
                dist->val[p] = 0;
        }
        sprintf(filename,"%s/DEBUG/geodesic/%s",img_parent_folder,file);
        iftWriteImageByExt(dist,filename);
    #endif

    iftDestroyImage(&dist);
    return(St);
}

void iftLabelLungsMarkers(iftImage *bin, iftSet *Si, iftLabeledSet **seeds){

    iftImage *L,*SetSi;
    iftAdjRel *A;
    iftVoxel u,v;
    iftSet *Q, *Saux, *S1, *S2;
    int *hist;
    int p, q, r, s, max, c[2], l;
    int n1, x1, l1;
    int n2, x2, l2;

    Q = Saux = S1 = S2 = NULL;

    // Creating Image Containing Set Si
    SetSi = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
    Saux = Si;
    while (Saux != NULL) {
        p = Saux->elem;
        SetSi->val[p] = 1;
        Saux = Saux->next;
    }

    // Labeling Connected Components
    L = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
    A = iftSpheric(ADJ_REL_RADIUS);
    Saux = Si;
    l = 1;
    while (Saux != NULL) {
        s = Saux->elem;
        if (L->val[s] == 0){
            L->val[s] = l;
            iftInsertSet(&Q,s);
            while (Q != NULL){
                p = iftRemoveSet(&Q);
                u = iftGetVoxelCoord(L,p);
                for (r = 1; r < A->n; r++){
                    v = iftGetAdjacentVoxel(A,u,r);
                    if (iftValidVoxel(L,v)) {
                        q = iftGetVoxelIndex(L, v);
                        if ((L->val[q] == 0) && (SetSi->val[q] == 1)) {
                            L->val[q] = L->val[p];
                            iftInsertSet(&Q, q);
                        }
                    }
                }
            }
            l++;
        }
        Saux = Saux->next;
    }
    iftDestroyImage(&SetSi);
    iftDestroyAdjRel(&A);

    // Creating Histogram from Labeled Image
    hist = iftAllocIntArray(l);
    for (p = 0; p < L->n; p++){
        if (L->val[p] != 0){
            hist[L->val[p]]++;
        }
    }

    // Selecting Labels of the two Largest Components (Left and Right Lungs)
    for (s = 0; s < 2; s++){
        max = 0;
        c[s] = -1;
        for (r = 0; r < l; r++){
            if (max < hist[r]){
                max = hist[r];
                c[s] = r;
            }
        }
        hist[c[s]] = 0;
    }
    iftFree(hist);

    // Separating Left and Right Lungs from Labeled Image and
    // Computing Center of Mass from them
    Saux = Si;
    x1 = x2 = n1 = n2 = 0;
    while (Saux != NULL) {
        p = Saux->elem;
        if ((L->val[p] == c[0]) && (c[0] != -1)) {
            iftInsertSet(&S1, p);
            u = iftGetVoxelCoord(bin, p);
            x1 += u.x; n1++;
        }
        else if ((L->val[p] == c[1]) && (c[0] != -1)) {
            iftInsertSet(&S2, p);
            u = iftGetVoxelCoord(bin, p);
            x2 += u.x; n2++;

        }
        Saux = Saux->next;
    }
    if (n1 != 0)
        x1 /= n1;
    if (n2 != 0)
        x2 /= n2;
    iftDestroyImage(&L);

    // Inserting Left and Right Lung on LabeledSet
    if (x1 < x2) {
        l1 = 1;
        l2 = 2;
    } else {
        l1 = 2;
        l2 = 1;
    }
    while (S1 != NULL){
        p = iftRemoveSet(&S1);
        iftInsertLabeledSet(seeds,p,l1);
    }
    while (S2 != NULL){
        p = iftRemoveSet(&S2);
        iftInsertLabeledSet(seeds,p,l2);
    }
}

void iftLabelTracheaMarkers(iftImage *bin, iftSet *St, iftLabeledSet **seeds){

    iftImage *L, *SetSt;
    iftAdjRel *A;
    iftVoxel u,v;
    iftSet *Q, *Saux;
    int *mean_z;
    int p, q, r, s;
    int l, t;
    int size, z;

    Q = Saux = NULL;

    // Creating Image Containing Set St
    SetSt = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
    Saux = St;
    size = 0;
    while (Saux != NULL) {
        p = Saux->elem;
        SetSt->val[p] = 1;
        Saux = Saux->next;
        size++;
    }

    // Labeling Connected Components and Computing their Center of Mass
    L = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
    A = iftSpheric(ADJ_REL_RADIUS);
    mean_z = iftAllocIntArray(size);
    Saux = St;
    l = 1;
    while (Saux != NULL) {
        s = Saux->elem;
        if (L->val[s] == 0){
            L->val[s] = l;
            iftInsertSet(&Q,s);
            size = 0;
            while (Q != NULL){
                p = iftRemoveSet(&Q);
                u = iftGetVoxelCoord(L,p);
                mean_z[L->val[p]] += u.z;
                size++;
                for (r = 1; r < A->n; r++){
                    v = iftGetAdjacentVoxel(A,u,r);
                    if (iftValidVoxel(L,v)) {
                        q = iftGetVoxelIndex(L, v);
                        if ((L->val[q] == 0) && (SetSt->val[q] == 1)) {
                            L->val[q] = L->val[p];
                            iftInsertSet(&Q, q);
                        }
                    }
                }
            }
            mean_z[l] /= size;
            l++;
        }
        Saux = Saux->next;
    }
    iftDestroyImage(&SetSt);
    iftDestroyAdjRel(&A);

    // Selecting the highest component
    z = 0;
    t = 0;
    for (p = 1; p <= l; p++){//l
        if (mean_z[p] > z){
            z = mean_z[p];
            t = p;
        }
    }
    iftFree(mean_z);

    // Inserting Trachea in LabeledSet
    int border;
    A = iftSpheric(1.0);
    for (p = 0; p < L->n; p++)
        if (L->val[p] == t){
            // Selecting seeds that do not belong to the border
            border = 0;
            u = iftGetVoxelCoord(L,p);
            for (int i = 1; i < A->n; i++) {
                v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(L, v)) {
                    q = iftGetVoxelIndex(L, v);
                    if (L->val[q] != t) {
                        border = 1;
                        break;
                    }
                }
            }
            if (!border)
                iftInsertLabeledSet(seeds,p,3);
        }
    iftDestroyImage(&L);
    iftDestroyAdjRel(&A);

}

iftLabeledSet *iftLabelMarkers(iftImage *bin, iftSet *Si, iftSet *St, iftSet *Se, iftSet **forbidden)
{
    int p;
    iftLabeledSet *seeds=NULL;
    iftSet *Saux = NULL;

    iftLabelLungsMarkers(bin,Si,&seeds);

    iftLabelTracheaMarkers(bin,St,&seeds);

    Saux = Se;
    while (Saux != NULL) {
        p = Saux->elem;
        iftInsertLabeledSet(&seeds,p,0);
        Saux = Saux->next;
    }

    *forbidden=NULL;
    for (int p=0; p < bin->n; p++)
        if (bin->val[p]==0)
            iftInsertSet(forbidden,p);

    return(seeds);
}