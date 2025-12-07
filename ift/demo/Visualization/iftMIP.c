//
// Created by azaelmsousa on 03/04/20.
//

#include "ift.h"

iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **img_out, int *spin, int *tilt);
void iftGetOptionalArgs(  iftDict *args, char **label_path);

iftColor iftGrayScaleToBlueToRedColor(float intensity, float norm_value)
{
    float value = 4*(intensity/norm_value)+1;

    iftColor rgb_color;
    rgb_color.val[0] = norm_value * iftMax(0,(3-(float)fabs(value-4)-(float)fabs(value-5))/2);
    rgb_color.val[1] = norm_value * iftMax(0,(4-(float)fabs(value-2)-(float)fabs(value-4))/2);
    rgb_color.val[2] = norm_value * iftMax(0,(3-(float)fabs(value-1)-(float)fabs(value-2))/2);

    iftColor ycbcr = iftRGBtoYCbCr(rgb_color, norm_value);

    return ycbcr;
}

iftImage *iftGrayImageToColorImage(iftImage *img, iftImage *mask){
    iftImage *colored_image;
    int Imax = iftNormalizationValue(iftMaximumValue(img));

    iftImage *aux = iftEqualize(img, Imax);
    colored_image = iftCreateColorImage(img->xsize, img->ysize, img->zsize,Imax);

    for(int p = 0; p < img->n; p++){
        if (mask ->val[p] == 0){
            iftSetRGB(colored_image,p,0,0,0,Imax);
        } else {
            iftColor ycbcr_color = iftGrayScaleToBlueToRedColor((float) aux->val[p], Imax);

            colored_image->val[p] = ycbcr_color.val[0];
            colored_image->Cb[p] = ycbcr_color.val[1];
            colored_image->Cr[p] = ycbcr_color.val[2];
        }
    }
    iftDestroyImage(&aux);

    return(colored_image);
}

iftIntArray *iftSListToIntArray(iftSList *SL)
{
    iftIntArray *arr = iftCreateIntArray(SL->n);

    iftSNode *node = SL->head;
    int i = 0;
    while (node != NULL) {
        arr->val[i++] = atoi(node->elem);
        node = node->next;
    }

    return arr;
}

float iftFImageValueAtPointOnMask(  iftFImage *img,   iftImage *mask, iftPoint P)
{
    iftVoxel u[8];
    int p[8], i;
    float dx,dy,dz, val[6], value;

    u[0].x = (int)P.x;      u[0].y = (int)P.y;       u[0].z = (int)P.z;
    u[1].x = u[0].x+1;      u[1].y = u[0].y;         u[1].z = u[0].z;
    u[2].x = u[0].x;        u[2].y = u[0].y + 1;     u[2].z = u[0].z;
    u[3].x = u[0].x+1;      u[3].y = u[0].y + 1;     u[3].z = u[0].z;
    u[4].x = u[0].x;        u[4].y = u[0].y;         u[4].z = u[0].z + 1;
    u[5].x = u[0].x+1;      u[5].y = u[0].y;         u[5].z = u[0].z + 1;
    u[6].x = u[0].x;        u[6].y = u[0].y + 1;     u[6].z = u[0].z + 1;
    u[7].x = u[0].x+1;      u[7].y = u[0].y + 1;     u[7].z = u[0].z + 1;

    for (i=0; i < 8; i++) {
        if (iftValidVoxel(img,u[i]))
            if (mask->val[iftGetVoxelIndex(mask,u[i])] > 0){
                p[i] = iftGetVoxelIndex(img, u[i]);
            } else {
                return 0;
            }
        else
            return 0;

    }

    dx = dy = dz = 1.0;

    val[0] =(float)img->val[p[1]]*dx+(float)img->val[p[0]]*(1.0-dx);
    val[1] =(float)img->val[p[3]]*dx+(float)img->val[p[2]]*(1.0-dx);
    val[2] =(float)img->val[p[5]]*dx+(float)img->val[p[4]]*(1.0-dx);
    val[3] =(float)img->val[p[7]]*dx+(float)img->val[p[6]]*(1.0-dx);
    val[4] = val[1]*dy + val[0]*(1.0-dy);
    val[5] = val[3]*dy + val[2]*(1.0-dy);
    value  = (val[5]*dz + val[4]*(1.0-dz));

    return(value);
}

int iftMaximumIntensityAlongRay(iftGraphicalContext *gc, iftPoint P0, iftPoint P1, iftPoint Pn, iftAdjRel *A)
{
    iftPoint     u;
    int          k, n, p, Imax;
    float        Dx=(Pn.x-P1.x), Dy=(Pn.y-P1.y), Dz=(Pn.z-P1.z);
    float        dx=0, dy=0, dz=0;
    iftVoxel     v;

    /* DDA - Digital Differential Analyzer */

    if (iftVectorsAreEqual(P1, Pn)) {
        n = 1;
    }else{ /* process points from P1 to Pn */
        if ((fabs(Dx) >= fabs(Dy))&&(fabs(Dx) >= fabs(Dz))) { /* Dx is the maximum projection of
  							     vector P1Pn */
            n  = (int)(fabs(Dx)+1);
            dx = iftSign(Dx);
            dy = dx*Dy/Dx;
            dz = dx*Dz/Dx;
        }else{
            if ((fabs(Dy) >= fabs(Dx))&&(fabs(Dy) >= fabs(Dz))) { /* Dy is the maximum projection of
  							       vector P1Pn */
                n  = (int)(fabs(Dy)+1);
                dy = iftSign(Dy);
                dx = dy*Dx/Dy;
                dz = dy*Dz/Dy;
            } else { /* Dz is the maximum projection of vector P1Pn */
                n  = (int)(fabs(Dz)+1);
                dz = iftSign(Dz);
                dx = dz*Dx/Dz;
                dy = dz*Dy/Dz;
            }
        }
    }

    /* Execute shading model along the viewing ray */

    u.x  = P1.x;  u.y = P1.y;  u.z = P1.z;
    Imax = 0;

    for (k=0; (k < n); k++) {

        v.x = iftRound(u.x);
        v.y = iftRound(u.y);
        v.z = iftRound(u.z);

        if (gc->label->val[iftFGetVoxelIndex(gc->scene, v)]!=0){
            /*for (i=0; (i < A->n)&&(flag==0); i++) { // it avoids skipping the surface of the objects
                w = iftGetAdjacentVoxel(A,v,i);
                if(iftFValidVoxel(gc->scene, w)){
                    p = iftFGetVoxelIndex(gc->scene,w); //v -> w
                        flag=1;
                        if (gc->object[gc->label->val[p]].visibility != 0){

                            int val = iftFImageValueAtPointOnMask(gc->scene,gc->label,u);
                            if (val > Imax)
                                Imax = val;
                        }
                }
            }*/
            p = iftFGetVoxelIndex(gc->scene,v); //v -> w
            if (gc->object[gc->label->val[p]].visibility != 0) {

                int val = iftFImageValueAtPointOnMask(gc->scene, gc->label, u);
                if (val > Imax)
                    Imax = val;
            }
        }

        u.x = u.x + dx;
        u.y = u.y + dy;
        u.z = u.z + dz;
    }

    return Imax;
}


iftImage *iftMaximumIntensityProjection(iftGraphicalContext *gc)
{
    iftImage  *image;
    iftVector  n;
    int        diag  = iftFDiagonalSize(gc->scene);
    iftAdjRel *A=iftSpheric(1.0);

    n.x  = 0; n.y = 0; n.z = 1;
    n    = iftTransformVector(gc->viewdir->Rinv,n);

    image =  iftCreateImage(diag,diag,1);

    for (int po=0; po < image->n; po++) {

        iftPoint  P0,P1,Pn;

        P0.x   =  iftGetXCoord(image,po);
        P0.y   =  iftGetYCoord(image,po);
        P0.z   = -diag/2.0;
        P0     =  iftTransformPoint(gc->viewdir->Tinv,P0);

        if (iftIntersectionPoints(gc,P0,n,&P1,&Pn)){
            image->val[po] = iftMaximumIntensityAlongRay(gc,P0,P1,Pn,A);
        }
    }

    iftDestroyAdjRel(&A);

    return image;
}


int main(int argc, const char *argv[]){

    iftDict *args = iftGetArgs(argc, argv);

    char *img_path     = NULL;
    char *img_out     = NULL;
    char *label_path   = NULL;
    timer *t1,*t2;
    int tilt,spin;

    iftGetRequiredArgs(args,&img_path,&img_out,&spin,&tilt);
    iftGetOptionalArgs(args,&label_path);

    // Initialization
    t1 = iftTic();

    puts("-- Initialization");
    iftImage *img = iftReadImageByExt(img_path);

    iftImage *label = NULL;
    if (label_path == NULL){
        label = iftCreateImageFromImage(img);
        for (int i = 0; i < label->n; i++) label->val[i] = 1;
    } else {
        label = iftReadImageByExt(label_path);
    }

    iftFImage *fimg = iftImageToFImage(img);
    iftDestroyImage(&img);

    // Creating Graphical Context
    puts("-- Creating Graphical Context");
    iftGraphicalContext *gc = iftCreateGraphicalContext(fimg,label);
    iftDestroyFImage(&fimg);
    iftSetViewDir(gc,tilt,spin);
    iftSetProjectionMode(gc,RAYCASTING);

    // Selecting visible objects
    char *objects = NULL;
    if ((label_path != NULL) && (iftDictContainKey("--skip-objs", args, NULL))) {
        objects = iftGetStrValFromDict("--skip-objs", args);
        iftSList *sl = iftSplitString(objects, ",");
        iftIntArray *arr = iftSListToIntArray(sl);
        int max_lb = iftMaximumValue(label);

        for(int lb = 1; lb <= max_lb; lb++)
            iftSetObjectVisibility(gc, lb, 1);
        for(int i = 0; i < arr->n; i++)
            iftSetObjectVisibility(gc, arr->val[i], 0);
    }
    iftDestroyImage(&label);

    // Maximum Intensity Projection
    puts("-- Maximum Intensity Projection");
    iftImage *mip = iftMaximumIntensityProjection(gc);

    iftImage *mip_mask = iftCreateImageFromImage(mip);
    for (int i = 0; i < mip->n; i++){
        if (mip->val[i] > 0){
            mip_mask->val[i] = 1;
        }
    }

    puts("-- Cropping image");
    iftVoxel v;
    iftBoundingBox bb = iftMinBoundingBox(mip,&v);
    iftPrintBoundingBox(bb);
    iftImage *out = iftCreateImage(iftMax(bb.end.x-bb.begin.x,1),iftMax(bb.end.y-bb.begin.y,1),iftMax(bb.end.z-bb.begin.z,1));
    iftImage *out_mask = iftCreateImage(iftMax(bb.end.x-bb.begin.x,1),iftMax(bb.end.y-bb.begin.y,1),iftMax(bb.end.z-bb.begin.z,1));

    int q = 0;
    int p;
    int z = bb.begin.z;
    do{
        int y = bb.begin.y;
        do{
            int x = bb.begin.x;
            do{
                v.x = x;
                v.y = y;
                v.z = z;

                if (iftValidVoxel(mip, v)) {
                    p = iftGetVoxelIndex(mip, v);
                    out->val[q] = mip->val[p];
                    out_mask->val[q] = mip_mask->val[p];
                }

                q++;
                x++;
            }while(x < bb.end.x);
            y++;CoFi
        }while(y < bb.end.y);
        z++;
    }while(z < bb.end.z);
    iftDestroyImage(&mip);
    iftDestroyImage(&mip_mask);
    mip_mask = out_mask;
    mip = out;
    out = NULL;
    out_mask = NULL;

    puts("-- Applying Normalization");
    int norm_value = 255;

    // Assuming the projection follows a normal distribution
    iftMatrix *M;
    M = iftMeanStdevValuesInRegions(mip,mip_mask,1);
    float min_border = iftMatrixElem(M,0,1)-3*iftMatrixElem(M,1,1);
    float max_border = iftMatrixElem(M,0,1)+3*iftMatrixElem(M,1,1);
    int acc_top = 0, acc_bottom = 0;
    for (int p = 0; p < mip->n; p++){
        if (mip->val[p] > max_border) {
            mip->val[p] = max_border;
            acc_top++;
        }
        if ((mip_mask->val[p] > 0) && (mip->val[p] < min_border)) {
            mip->val[p] = min_border;
            acc_bottom++;
        }
    }

    iftDestroyMatrix(&M);

    iftImage *norm = iftNormalizeInRegion(mip,mip_mask,0,norm_value);
    iftDestroyImage(&mip);

    if (iftDictContainKey("--apply-colormap", args, NULL)) {

        iftImage *color = iftCreateColorImage(norm->xsize, norm->ysize, norm->zsize, iftImageDepth(norm));
        iftColorTable *ctb = iftHeatMapColorTable(norm_value);
        for (int p = 0; p < color->n; p++) {
            if (mip_mask->val[p] > 0) {
                iftSetYCbCr(color,p,ctb->color[norm->val[p]]);
            }else
                iftSetRGB(color, p, 0, 0, 0, norm_value);
        }

        //iftImage *color = iftGrayImageToColorImage(norm,mip_mask);

        iftDestroyImage(&norm);
        norm = iftCopyImage(color);
        iftDestroyImage(&color);

    }
    iftDestroyImage(&mip_mask);

    iftWriteImageByExt(norm,img_out);
    iftDestroyImage(&norm);
    iftDestroyGraphicalContext(gc);

    t2 = iftToc();

    fprintf(stdout, "Done...\nMaximum Intensity Projection took %s\n", iftFormattedTime(iftCompTime(t1, t2)));

    return 0;

}


iftDict *iftGetArgs(int argc, const char *argv[])
{
    char program_description[2048] = \
        "- Maximum Intensity Projection (MIP).\n";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input Scene Image."},
            {.short_name = "-s", .long_name = "--spin", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=true, .help="Rotation spin."},
            {.short_name = "-t", .long_name = "--tilt", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                    .required=true, .help="Rotation tilt."},
            {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output image."},
            {.short_name = "-l", .long_name = "--label-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Label image. MIP will be performed only inside mask."},
            {.short_name = "-a", .long_name = "--skip-objs", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=false, .help="Object label to be skipped from the projection (e.g. \"1,2,3\"). If not given, all objects will be considered."},
            {.short_name = "-c", .long_name = "--apply-colormap", .has_arg=false, .arg_type=IFT_UNTYPED,
                    .required=false, .help="Apply a colormap before output the mip image"},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(  iftDict *args, char **img_path, char **img_out, int *spin, int *tilt)
{
    *img_path     = iftGetStrValFromDict("--input-img", args);
    *img_out      = iftGetStrValFromDict("--output-img", args);
    *spin         = iftGetLongValFromDict("--spin",args);
    *tilt         = iftGetLongValFromDict("--tilt",args);

    puts("");
    puts("--------------------------------------");
    printf("- Input Image: \"%s\"\n", *img_path);
    printf("- Tilt: %d\n", *tilt);
    printf("- Spin: %d\n", *spin);
    printf("- Output Image: \"%s\"\n", *img_out);
    puts("--------------------------------------");
    puts("");
}

void iftGetOptionalArgs(  iftDict *args, char **label_path)
{
    if (iftDictContainKey("--label-img", args, NULL))
        *label_path = iftGetStrValFromDict("--label-img", args);
}