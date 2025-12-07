#include "ift.h"

iftDict *iftGetArguments(int argc, const char *argv[]);

int main(int argc, const char *argv[])
{
    iftImage  *img,*label,*rend;
    iftFImage *scene;
    iftGraphicalContext *gc;
    timer     *t1=NULL,*t2=NULL;
    float      tilt, spin, opacity;
    iftColorTable *cmap = NULL;
    int max_lb;
    iftDict *args = NULL;
    iftSList *opacity_values = NULL;
    iftSNode *opacity_node = NULL;
    int projection_mode = RAYCASTING;

    /*--------------------------------------------------------*/
    size_t MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/

    args = iftGetArguments(argc, argv);

    img    = iftReadImageByExt(iftGetConstStrValFromDict("--input", args));
    label  = iftReadImageByExt(iftGetConstStrValFromDict("--label", args));
    scene  = iftImageToFImage(img);
    iftDestroyImage(&img);

    opacity_values = iftSplitString(iftGetConstStrValFromDict("--object-opacity", args), " ");
//    opacity = (float)iftGetDblValFromDict("--object-opacity", args);

    if(iftDictContainKey("--projection-mode", args, NULL)) {
        if(iftCompareStrings(iftGetConstStrValFromDict("--projection-mode", args),"RAYCASTING")) {
            projection_mode = RAYCASTING;
        } else {
            projection_mode = SPLATTING;
        }
    }

    if (iftDictContainKey("--tilt", args, NULL)){ // Tilt and Spin must align principal axis with the z axis
        if(iftDictContainKey("--spin", args, NULL)) {
            tilt = (float)iftGetDblValFromDict("--tilt", args);
            spin = (float)iftGetDblValFromDict("--spin", args);
        } else {
            iftError("Tilt and spin angle must be provided together", "main");
        }

    } else if(!iftDictContainKey("--spin", args, NULL)){
        iftTiltSpinFromPrincipalAxis(label, &tilt, &spin);
    } else {
        iftError("Tilt and spin angle must be provided together", "main");
    }

    t1  = iftTic();

    max_lb  = iftMaximumValue(label);
    if(max_lb >= 6 && iftCompareStrings(iftGetConstStrValFromDict("--colormap", args), "JET")) {
        cmap = iftBlueToRedColorTable(max_lb + 1);
    } else {
        if(iftCompareStrings(iftGetConstStrValFromDict("--colormap", args), "JET"))
            iftWarning("Less than 7 colors in the label map, using RANDOM colormap instead", "main");

        cmap = iftCreateColorTable(max_lb + 1);
    }


    t2  = iftToc();

    fprintf(stdout,"Colormap creation in %f ms\n",iftCompTime(t1,t2));

    t1 = iftTic();

    gc    = iftCreateGraphicalContext(scene,label);
    iftSetViewDir(gc,tilt,spin);
    if(iftCompareStrings(iftGetConstStrValFromDict("--object-normal-type", args), "SCENE")) {
        iftSetObjectNormal(gc, SCENE_NORMAL);
    } else {
        iftSetObjectNormal(gc, OBJECT_NORMAL);
    }

    opacity_node = opacity_values->head;

    for(int lb = 1; lb <= max_lb; lb++) {
        iftFColor rgb;
        rgb.val[0] = (float)(cmap->color[lb].val[0] / 255.0);
        rgb.val[1] = (float)(cmap->color[lb].val[1] / 255.0);
        rgb.val[2] = (float)(cmap->color[lb].val[2] / 255.0);

        opacity = (float)atof(opacity_node->elem);
        if(opacity_values->n > 1 && opacity_values->n == max_lb) {
            opacity_node = opacity_node->next;
        }

        iftSetObjectColor(gc, lb, rgb.val[0], rgb.val[1], rgb.val[2]);
        iftSetObjectVisibility(gc, lb, 1);
        iftSetObjectOpacity(gc, lb, opacity);
    }

    iftSetProjectionMode(gc,projection_mode);

    t2     = iftToc();

    fprintf(stdout,"Preprocessing in %f ms\n",iftCompTime(t1,t2));

    t1 = iftTic();

    rend = iftSurfaceRender(gc);
    
    t2 = iftToc();

    fprintf(stdout, "Rendering in %f ms\n", iftCompTime(t1, t2));
    img = iftNormalize(rend, 0, 255);

    iftWriteImageByExt(img, iftGetConstStrValFromDict("--output", args));

    iftDestroyImage(&label);
    iftDestroyFImage(&scene);
    iftDestroyGraphicalContext(gc);

    iftDestroyImage(&rend);
    iftDestroyImage(&img);
    iftDestroySList(&opacity_values);



    iftDestroyColorTable(&cmap);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%lu, %lu)\n",
               MemDinInicial,MemDinFinal);

    return(0);
}


iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = "This is a program renders the objects in a scene";

    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input image (3D .scn)"},
            {.short_name = "-l", .long_name = "--label", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input label (3D .scn)"},
            {.short_name = "-o", .long_name = "--output", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output image (.ppm, .png, .jpeg)"},
            {.short_name = "-a", .long_name = "--object-opacity", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Opacity values for the objects within [0.0, 1.0]. Either a single common opacity must be given for all objects, or one for each object must be specified. Example \"0.1 0.2 0.5\" or \"0.3\" for all."},
            {.short_name = "-c", .long_name = "--colormap", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Colormap to be used {RANDOM, JET}"},
            {.short_name = "-t", .long_name = "--object-normal-type", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Object normal calculation type {SCENE, OBJECT}"},
            {.short_name = "-m", .long_name = "--projection-mode", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Projection mode {RAYCASTING, SPLATTING}. Default is RAYCASTING"},
            {.short_name = "", .long_name = "--tilt", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=false, .help="Tilt angle"},
            {.short_name = "", .long_name = "--spin", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=false, .help="Spin angle"}
    };
    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}
