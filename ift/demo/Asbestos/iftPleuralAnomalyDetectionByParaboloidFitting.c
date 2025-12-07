//
// Created by azaelmsousa on 26/08/21.
//

#include "ift.h"

#define LEFT_LUNG 1
#define RIGHT_LUNG 2

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **img_out);


iftImage *iftImproveALTIS(iftImage *segm);

/*    Curve Fitting */
iftFloatArray *iftParaboloidFitting(iftImage *bin, int nepochs, float learning_rate);
char iftParaboloidSide(iftFloatArray *theta, iftPoint p, iftPoint center);
/*************************************************************/

int main(int argc, const char *argv[])
{
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path          = NULL;
    char *out_path          = NULL;

    iftGetRequiredArgs(args,&img_path,&out_path);

    iftImage *label = iftReadImageByExt(img_path);

    puts("- Applying pleural anomaly detection");
    iftImage *out = iftImproveALTIS(label);
    iftDestroyImage(&label);

    puts("- Writing output image");
    iftWriteImageByExt(out,out_path);

    puts("- Deallocating memory");
    iftDestroyImage(&out);

    return 0;
}

iftDict *iftGetArgs(int argc, const char *argv[])
{
    char program_description[2048] = \
        "- This program applies 2D skeleton operator to separate the pleura costal and mediastinal." \
        "From this point, the anomalies located at the costal pleura are detected by morphological operations\n";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-l", .long_name = "--label-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input Scene Image."},
            {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output Scene Image."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

void iftGetRequiredArgs(  iftDict *args, char **img_path, char **out_img) {
    *img_path = iftGetStrValFromDict("--label-img", args);
    *out_img = iftGetStrValFromDict("--output-img", args);

    if (!iftFileExists(*img_path))
        iftError("Input image %s does not exists", "iftGetRequiredArgs", *img_path);

    puts("-----------------------");
    printf("- Label Image: %s\n", *img_path);
    printf("- Output Image: %s\n", *out_img);
    puts("-----------------------");
}

iftImage *iftImproveALTIS(iftImage *segm)
{
    puts("--- Left Lung");
    iftImage *LL = iftExtractObject(segm,LEFT_LUNG);
    iftSet *seeds = NULL; //
    iftImage *lung_dil = iftDilateBin(LL,&seeds,20.); //
    iftDestroySet(&seeds);

    puts("----- Skel");
    iftAdjRel *A = iftSpheric(sqrt(3.));
    iftFImage *geo_skel = iftMSSkel(lung_dil);
    float thres = 0.1;
    float maxval = iftFMaximumValue(geo_skel);
    iftImage *lung_skel = iftSurfaceSkeleton(geo_skel,thres*maxval,1);
    iftDestroyFImage(&geo_skel);
    iftDestroyAdjRel(&A);
    iftDestroyImage(&lung_dil);

    puts("----- Fitting paraboloid");
    iftFloatArray *theta = iftParaboloidFitting(lung_skel,100,0.1);

    puts("----- Printing curve");
    iftImage *curve = iftCreateImageFromImage(segm);
    iftPoint center = iftGeometricCenter(lung_skel);
    for (int i = 0; i < curve->n; i++){
        iftPoint p;
        iftVoxel u = iftGetVoxelCoord(curve,i);
        p.x = u.x;
        p.y = u.y;
        p.z = u.z;
        char side = iftParaboloidSide(theta,p,center);
        if (side == -1)
            curve->val[i] = 2;
        else if (side == 1)
            curve->val[i] = 1;
    }
    iftWriteImageByExt(curve,"curve.nii.gz");
    iftDestroyImage(&lung_skel);
    iftDestroyFloatArray(&theta);
    iftDestroyImage(&LL);


    return iftCopyImage(segm);

}

/* Stochastic Gradient Descent (SGD)
 *
 * (x-i)^2   (y-j)^2   (z-k)^2
 * ------- + ------- + ------- = 1
 *   a^2       b^2       c^2
 *
 * F'(a,b,c)        (x-i)^2
 * --------- = -2 * -------
 *     a'             a^3
 *
 *            1
 * L(pred) = ---- * sum (y - F(a,b,c))^2
 *            2N
 *                       _                      _
 * L'(pred)   -2        |              (x-i)^2   |
 * -------- = --- * sum | (y - pred) * -------   |
 *    a'       N        |_               a^3    _|
 */

float F(float *theta, iftPoint point, iftPoint center)
{
    return (-(point.x-center.x))/(theta[0]) +
           ((point.y-center.y)*(point.y-center.y))/(theta[1]*theta[1]) +
           ((point.z-center.z)*(point.z-center.z))/(theta[2]*theta[2]);
}

float dLdx(float pred,float theta, float point_coord, float center_coord)
{
    return (1-pred)*(-(point_coord-center_coord)/(theta*theta));
}

float dLdy(float pred,float theta, float point_coord, float center_coord)
{
    return (1-pred)*((point_coord-center_coord)*(point_coord-center_coord)/(theta*theta*theta));
}

float dLdz(float pred,float theta, float point_coord, float center_coord)
{
    return (1-pred)*((point_coord-center_coord)*(point_coord-center_coord)/(theta*theta*theta));
}

iftFloatArray *iftParaboloidFitting(iftImage *bin, int nepochs, float learning_rate)
{
    // Initialization
    iftFloatArray *theta = iftCreateFloatArray(3);
    theta->val[0] = 10.;
    theta->val[1] = 10.;
    theta->val[2] = 10.;
    iftPoint center = iftGeometricCenter(bin), point;
    iftDataSet *Z = iftObjectToDataSet(bin);
    float pred;

    // SGD
    for (int e = 0; e < nepochs; e++){
        printf("%f,%f,%f\n",theta->val[0],theta->val[1],theta->val[2]);
        float loss = 0;
        // stack gradients
        iftPoint grad;
        grad.x = 0;
        grad.y = 0;
        grad.z = 0;
        for (int s = 0; s < Z->nsamples; s++){
            point.x = Z->sample[s].feat[0];
            point.y = Z->sample[s].feat[1];
            point.z = Z->sample[s].feat[2];
            pred = F(theta->val,point,center);
            grad.x += dLdx(pred,theta->val[0],point.x,center.x);
            grad.y += dLdy(pred,theta->val[1],point.y,center.y);
            grad.z += dLdz(pred,theta->val[2],point.z,center.z);
            loss += 1-pred;
        }
        grad.x = 2*grad.x/Z->nsamples;
        grad.y = 2*grad.y/Z->nsamples;
        grad.z = 1*grad.z/Z->nsamples;
        loss /= 2*Z->nsamples;
        // update theta
        theta->val[0] = theta->val[0] - learning_rate*grad.x;
        theta->val[1] = theta->val[1] - learning_rate*grad.y;
        theta->val[2] = theta->val[2] - learning_rate*grad.z;
        printf("%f,%f,%f\n",grad.x,grad.y,grad.z);
        printf("%.4f\n",loss);
        puts("-----");
    }

    iftDestroyDataSet(&Z);
    return theta;
}

char iftParaboloidSide(iftFloatArray *theta, iftPoint p, iftPoint center)
{
    float pred = F(theta->val,p,center);

    if (pred > 1)
        return 1;
    else if (pred < 1)
        return -1;
    else
        return 0;
}