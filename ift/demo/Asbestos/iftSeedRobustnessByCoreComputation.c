//
// Created by azael on 27/02/19.
//

#include "ift.h"

/********************* DICTIONARY HEADERS *********************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args,char **img_path, char **segm_path, char **seeds_path);
void iftValidateRequiredImgArgs(const char *img_path);
void iftValidateRequiredTxtArgs(const char *seeds_path);
/**************************************************************/

/********************** FUNCTION HEADERS **********************/
void iftComputeCore(  iftImage *grad, iftImage *obj, iftImage **core, iftAdjRel *A);
iftImage *iftFindCores(  iftImage *grad,   iftImage *segm);
double iftSeedRobustness(iftImage *core,iftLabeledSet *seeds);
/**************************************************************/

/************************* CONSTANTS **************************/
#define CORE_OVERLAP_THRESHOLD 0.85
#define CORE_VOLUME_THRESHOLD_FACTOR 0.15
/**************************************************************/

int main(int argc, const char *argv[])
{
    iftDict *args = iftGetArgs(argc,argv);
    char *img_path = NULL;
    char *seeds_path = NULL;
    char *segm_path = NULL;
    double score = 0.;

    iftGetRequiredArgs(args,&img_path,&segm_path,&seeds_path);

    iftImage *grad = iftReadImageByExt(img_path);
    iftImage *segm = iftReadImageByExt(segm_path);
    iftLabeledSet *l = iftReadSeeds(grad,seeds_path);

    puts("- Finding Cores");
    iftImage *cores = iftFindCores(grad,segm);

    puts("- Computing Seed Robustness");
    score = iftSeedRobustness(cores,l);
    printf("--- Seed Robustness: %.2f\n",score);

    puts("- Dealocating memory");
    iftDestroyImage(&grad);
    iftDestroyImage(&segm);
    iftDestroyImage(&cores);
    iftDestroyLabeledSet(&l);

    return 0;
}

/********************* DICTIONARY FUNCTIONS *********************/

iftDict *iftGetArgs(int argc, const char *argv[])
{
    char program_description[2048] = \
        "- This program evaluates the robustness of estimated seeds by computing the core of the IFT.\n" \
        "- You can pass the image where the segmentation would be performed and,\n" \
        "- The file containing the seeds used for segmentation.\n\n" \
        " This implementation is the based on the paper: Seed Robustness of Oriented Image Foresting Transform: Core Computation and the Robustness Coefficient\n" \
        " TAVARES, Anderson Carlos Moreira; BEJAR, Hans Harley Ccacyahuillca; MIRANDA, Paulo André Vechiatto, 2017.";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input image. The minimum spanning forest is computed over it."},
            {.short_name = "-l", .long_name = "--label-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Label image with the objects of interest segmented."},
            {.short_name = "-s", .long_name = "--seeds", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Objects and background seeds."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

void iftGetRequiredArgs(  iftDict *args,char **img_path, char **segm_path, char **seeds_path)
{
    *img_path       = iftGetStrValFromDict("--input-img", args);
    *segm_path       = iftGetStrValFromDict("--label-img", args);
    *seeds_path      = iftGetStrValFromDict("--seeds", args);

    iftValidateRequiredImgArgs(*img_path);
    iftValidateRequiredImgArgs(*segm_path);
    iftValidateRequiredTxtArgs(*seeds_path);

    puts("");
    puts("--------------------------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Label Image: \"%s\"\n", *segm_path);
    printf("- Seeds: \"%s\"\n", *seeds_path);
    puts("--------------------------------------");
    puts("");
}

void iftValidateRequiredImgArgs(const char *img_path)
{
    if (!iftIsImageFile(img_path))
        iftError("Invalid Input Image: \"%s\"", "iftValidateRequiredImgArgs", img_path);
}

void iftValidateRequiredTxtArgs(const char *seeds_path)
{
    if (!iftFileExists(seeds_path))
        iftError("Invalid Input Seeds: \"%s\"", "iftValidateRequiredTxtArgs", seeds_path);
}

/************************** FUNCTIONS **************************/

iftImage *iftFindCores(  iftImage *grad,   iftImage *segm)
{
    iftImage *core = iftCreateImageFromImage(grad);
    iftImage *close = NULL;
    iftAdjRel *A = iftSpheric(sqrt(3.0));
    iftImage *obj = NULL;
    int labels = iftMaximumValue(segm);

    /* The cores of each object is computed separately.
     * The reason for that is that the implementation
     * ought to be simpler, is easier to debug and it
     * avoids huge and complex methods.*/
    for (int i = 1; i <= labels; i++){
        printf("=========== Object %d ===========\n",i);
        obj = iftExtractObject(segm,i);
        iftComputeCore(grad,obj,&core,A);
        iftDestroyImage(&obj);
        /* Mapping small cores as background*/
        for (int i = 0; i < core->n; i++)
            if (core->val[i] == -1)
                core->val[i] = 0;	
    }

    close = iftCloseBasins(core,NULL,NULL);
    for( int i = 0; i < close->n; i++)
	core->val[i] = close->val[i];
    iftDestroyImage(&close);    

    puts("================================");

    iftWriteImageByExt(core,"cores.scn");

    iftDestroyAdjRel(&A);
    return core;
}


void iftComputeCore(  iftImage *grad, iftImage *obj, iftImage **core, iftAdjRel *A)
{
    iftImage *cost  = NULL;
    iftImage *th    = NULL;
    iftGQueue *Q    = NULL;
    int p, core_id, leaking_value, leaking_point;
    int overlapping, total, pos;
    //iftPoint po;
    iftVoxel u,v;
    char first_leak;
    double score = 0.;

    pos = 0;
    while (score < CORE_OVERLAP_THRESHOLD) {

        /* Initialization */

        cost = iftCreateImage(grad->xsize, grad->ysize, grad->xsize);
        Q = iftCreateGQueue(iftMaximumValue(grad) + 1, grad->n, cost->val);

        /* Selecting a core point. Initially, the core point is the
         * geometric center. If this point already belongs to a
         * core, the first available point (not a core and inside the
         * object) is selected.*/
        puts("--- Detecting Available Seed");	
        //po = iftGeometricCenter(obj);
        //u.x = (int) po.x;
        //u.y = (int) po.y;
        //u.z = (int) po.z;
        p = pos++;
        /* I will keep searching for a voxel which does not already
         * belong to a core and is inside the object*/
        while (((*core)->val[p] != 0) || (obj->val[p] == 0)){
            //if ((obj->val[pos] > 0) && ((*core)->val[pos] == 0))		
                p = pos;
		pos++;
            //else
                //pos++;
            if (pos >= obj->n) {
                /* Dealocating memory */
                iftDestroyGQueue(&Q);
                iftDestroyImage(&cost);
                return;
            }
        }

        for (int i = 0; i < cost->n; i++) {
            cost->val[i] = IFT_INFINITY_INT;
        }

        cost->val[p] = 0;
        iftInsertGQueue(&Q, p);
        first_leak = 1;
        leaking_value = 0;
        leaking_point = p;

        /* Image Foresting Transform */

        puts("--- Image Foresting Transform");
        while (!iftEmptyGQueue(Q)) {
            p = iftRemoveGQueue(Q);
            u = iftGetVoxelCoord(grad, p);

            for (int i = 1; i < A->n; i++) {
                v = iftGetAdjacentVoxel(A, u, i);
                if (iftValidVoxel(grad, v)) {
                    int q = iftGetVoxelIndex(grad, v);
                    /* If the label of the adjacent voxel is 0, then it is a
                     * leak and since it is the first one to be out of the Queue,
                     * it implies that it is the leak with lowest cost.*/
                    if ((obj->val[q] == 0) && (first_leak)) {
                        first_leak = 0;
                        leaking_point = p;
                        leaking_value = iftMax(grad->val[q], grad->val[p]);
                    } else
                        /* On the other hand, the IFT only propagates to voxels that
                         * have not been associated with a core yet.*/
                    if ((Q->L.elem[q].color != IFT_BLACK) && ((*core)->val[q] == 0) && (obj->val[q] > 0)) {
                        int arc_w = iftMax(cost->val[p],grad->val[q]);//iftMax(grad->val[q], grad->val[p]);
                        if (arc_w < cost->val[q]) {
                            if (Q->L.elem[q].color == IFT_GRAY)
                                iftRemoveGQueueElem(Q, q);
                            cost->val[q] = arc_w;
                            iftInsertGQueue(&Q, q);
                        }
                    }
                }
            }
        }


        /* Once the IFT is complete, we apply a threshold on
         * the cost map according to the leaking value. A voxel
         * is said to belong to the core if its cost is less
         * than the leaking value and if it does not belong
         * to any other core.*/
        puts("--- Thresholding Cost Map");
        th = iftCreateImage(grad->xsize, grad->ysize, grad->xsize);
        th->val[leaking_point] = 1;
        for (int i = 0; i < cost->n; i++) {
            if ((cost->val[i] < leaking_value) && (obj->val[i] > 0) && ((*core)->val[i] == 0))
                th->val[i] = 1;
        }

        /* The threshold may leave unconnected components.
         * Thus, the core is the connected component containing
         * the leakeage.*/
        core_id = iftMaximumValue(*core) + 1;
        iftImage *comps = iftLabelComp(th,A);
	iftDestroyImage(&th);
	th = iftCreateImageFromImage(comps);
        int comp_label = comps->val[leaking_point];
        for (int i = 0; i < comps->n; i++){
            if (comps->val[i] == comp_label){
                th->val[i] = core_id;
		(*core)->val[i] = core_id; //teste
	    }
        }
        iftDestroyImage(&comps);        

        /* After the threshold is performed, some holes
         * inside the core may occur. Therefore, we end
         * the process with the application of a Close
         * Basins operation.*/
	/*
        puts("--- Close Basins");
        iftImage *close = iftCloseBasins(th, NULL, NULL);
	iftDestroyImage(&th);
        for (int i = 0; i < close->n; i++) {
            if (close->val[i] > 0)
                (*core)->val[i] = core_id;
        }*/
	iftImage *close = iftCopyImage(th);
	iftDestroyImage(&th);
	
        /* The core may be really small and it must be
         * removed.*/
        int volume = 0;
        int vol_th = (int)(CORE_VOLUME_THRESHOLD_FACTOR*iftAreaVolumeOfObject(obj,iftMaximumValue(obj)));
        for (int i = 0; i < (*core)->n; i++)
            if ((*core)->val[i] == core_id)
                volume++;
        printf("--- Core %d has volume: %d. Minimum of %d\n",core_id,volume,vol_th);
        if (volume < vol_th){
            for (int i = 0; i < (*core)->n; i++)
                if ((*core)->val[i] == core_id)
                    (*core)->val[i] = -1;
            puts("Core is too small, selecting another seed.");
            puts("-------------");

            /* Dealocating memory */
            iftDestroyGQueue(&Q);
            iftDestroyImage(&cost);
            iftDestroyImage(&close);
            continue;
        }

        /* Computing score to see if another IFT is
         * required. This score is based on the
         * percentage of overlapping between the cores
         * and the original object.*/
        score = 0.;
        overlapping = total = 0;
        for (int i = 0; i < close->n; i++) {
            if ((close->val[i] > 0) && (obj->val[i] > 0))
                overlapping++;
            if (obj->val[i] > 0)
                total++;
        }
        score = (float) overlapping / total;
        iftDestroyImage(&close);
        printf("--- Core overlapping of: %f\n",score);
        puts("-------------");

        /* Dealocating memory */
        iftDestroyGQueue(&Q);
        iftDestroyImage(&cost);
    }
}

double iftSeedRobustness(iftImage *core,iftLabeledSet *seeds)
{
    double score = 0.;
    int max = iftMaximumValue(core);
    int cores_intersected = 0;
    iftIntArray *intersection = iftCreateIntArray(max+1);
    iftLabeledSet *Saux = NULL;

    Saux = seeds;
    while (Saux != NULL){
        int label = Saux->label;
        intersection->val[label]++;
        Saux = Saux->next;
    }

    for (int i = 1; i <= intersection->n; i++)
        if (intersection->val[i] > 0)
            cores_intersected++;
    iftDestroyIntArray(&intersection);

    score = (double)cores_intersected/max;

    return score;
}

/***************************************************************/

