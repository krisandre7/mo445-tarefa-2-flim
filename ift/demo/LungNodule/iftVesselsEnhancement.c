/*
 * Created by Azael M. Sousa on 04/05/21.
 *
 *
 */

#include "ift.h"

//************************* ARGUMENTS ************************
iftDict *iftGetArguments(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **input_img_path, char **out_path, float *radius, float *stdev_start, float *stdev_end,
                        float *stdev_step, float *alpha, float *beta, float *gamma);
//*************************************************************


int main(int argc, const char *argv[])
{

    iftDict *args = iftGetArguments(argc, argv);

    // required args
    char *img_path    = NULL;
    char *out_path    = NULL;
    int n_cols        = 3;
    int n_rows        = 3;
    float radius,stdev_start,stdev_end,stdev_step,alpha,beta,gamma;

    iftGetRequiredArgs(args, &img_path, &out_path,&radius,&stdev_start,&stdev_end,&stdev_step,&alpha,&beta,&gamma);

    puts("--- Reading data");
    iftImage *input_img    = iftReadImageByExt(img_path);
    iftFImage *vi          = iftCreateFImage(input_img->xsize,input_img->ysize,input_img->zsize); // probability of a voxel be a vessel

    while (stdev_start <= stdev_end) {

        printf("--- Hessian Filter at stdev: %.2f\n",stdev_start);

        puts("----- Computing Hessian Matrix");
        iftHessianImage *H = iftHessianImageByGaussian(input_img, radius, stdev_start);

        puts("----- Applying filter");
        for (int i = 0; i < input_img->n; i++) {
            iftMatrix *local_hessian = iftCreateMatrix(n_cols, n_rows);
            iftMatrix *S;

            iftMatrixElem(local_hessian, 0, 0) = H->Dxx->val[i];
            iftMatrixElem(local_hessian, 0, 1) = H->Dxy->val[i];
            iftMatrixElem(local_hessian, 0, 2) = H->Dxz->val[i];
            iftMatrixElem(local_hessian, 1, 0) = H->Dxy->val[i];
            iftMatrixElem(local_hessian, 1, 1) = H->Dyy->val[i];
            iftMatrixElem(local_hessian, 1, 2) = H->Dyz->val[i];
            iftMatrixElem(local_hessian, 2, 0) = H->Dxz->val[i];
            iftMatrixElem(local_hessian, 2, 1) = H->Dyz->val[i];
            iftMatrixElem(local_hessian, 2, 2) = H->Dzz->val[i];

            S = iftGeneralizedEigenvalues(local_hessian);

            iftMatrix *aux = iftCopyMatrix(S);
            for (int c = 0; c < aux->ncols; c++)
                for (int r = 0; r < aux->nrows; r++)
                    iftMatrixElem(aux, c, r) = fabs(iftMatrixElem(aux, c, r));
            int *index = (int *) calloc(sizeof(int), 3);
            index[0] = 0;
            index[1] = 1;
            index[2] = 2;
            iftFQuickSort(aux->val, index, 0, 2, IFT_DECREASING);
            float abs_lambda3 = iftMatrixElem(aux, 0, 0);
            float lambda3 = iftMatrixElem(S, 0, index[0]);
            float abs_lambda2 = iftMatrixElem(aux, 0, 1);
            float lambda2 = iftMatrixElem(S, 0, index[1]);
            float abs_lambda1 = iftMatrixElem(aux, 0, 2);
            float lambda1 = iftMatrixElem(S, 0, index[2]);
            iftDestroyMatrix(&aux);
            free(index);
            float vo;

            if ((lambda2 > 0) || (lambda3 > 0)) {
                vo = 0.;
            } else {
                float Rb = abs_lambda1 / sqrt((abs_lambda2 * abs_lambda3));
                float Ra = abs_lambda2 / abs_lambda3;
                float S2 = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
                float new_gamma = gamma * sqrt(S2);
                float vo_part1 = (1 - exp(-(Ra * Ra) / (2 * alpha * alpha)));
                float vo_part2 = exp(-(Rb * Rb) / (2 * beta * beta));
                float vo_part3 = (1 - exp(-S2 / (2 * new_gamma * new_gamma)));
                vo = vo_part1 * vo_part2 * vo_part3;
            }
            if (vo > vi->val[i]) {
                vi->val[i] = vo;
            }

            iftDestroyMatrix(&S);
            iftDestroyMatrix(&local_hessian);
        }
        iftDestroyHessianImage(&H);

        stdev_start += stdev_step;
    }
    iftImage *vessels = iftFImageToImage(vi,100);

    /*
    iftFImage *enhanced_vessels = iftMultByFImage(input_img,vi);
    iftImage *vessels = iftFImageToImage(enhanced_vessels,iftMaximumValue(input_img));
    iftDestroyFImage(&enhanced_vessels);
    iftDestroyFImage(&vi);*/

    puts("--- Dealocating memory");
    iftDestroyImage(&input_img);

    puts("--- Writing output image");
    iftWriteImageByExt(vessels,out_path);
    iftDestroyImage(&vessels);

    return 0;
}

//************************** SOURCES **************************
iftDict *iftGetArguments(int argc, const char *argv[]) {
    char program_description[2048] = \
        "This program applies a top-hat morphological transformation based on the Hessian Filter to find the tubular shaped structures.\n" \
        "It computes the transformation for a range of standard deviation values. The user must enter the first and final stdev values, " \
        "together with the range step.";


    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input CT image."},
            {.short_name = "-o", .long_name = "--output-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output filtered image."},
            {.short_name = "-r", .long_name = "--gaussian-radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Radius for the Gaussian kernel."},
            {.short_name = "-si", .long_name = "--gaussian-stdev-start", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="First standard deviation for the gaussian kernel."},
            {.short_name = "-sf", .long_name = "--gaussian-stdev-end", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Final standard deviation for the gaussian kernel."},
            {.short_name = "-st", .long_name = "--gaussian-stdev-step", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Step of the standard deviation."},
            {.short_name = "-a", .long_name = "--alpha", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Threshold for blob-like structures [0,1]."},
            {.short_name = "-b", .long_name = "--beta", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Threshold to distinguish between plate-like and line-like structures [0,1]."},
            {.short_name = "-c", .long_name = "--gamma", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=true, .help="Threshold to reduce the response of background pixels [0,1]."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

void iftGetRequiredArgs(  iftDict *args, char **input_img_path, char **out_path, float *radius, float *stdev_start, float *stdev_end,
                        float *stdev_step, float *alpha, float *beta, float *gamma) {
    *input_img_path                = iftGetStrValFromDict("--input-image", args);
    *out_path                      = iftGetStrValFromDict("--output-image", args);
    *radius                        = iftGetDblValFromDict("--gaussian-radius", args);
    *stdev_start                   = iftGetDblValFromDict("--gaussian-stdev-start", args);
    *stdev_end                     = iftGetDblValFromDict("--gaussian-stdev-end", args);
    *stdev_step                    = iftGetDblValFromDict("--gaussian-stdev-step", args);
    *alpha                         = iftGetDblValFromDict("--alpha", args);
    *beta                          = iftGetDblValFromDict("--beta", args);
    *gamma                         = iftGetDblValFromDict("--gamma", args);

    puts("-----------------------");
    printf("- Input Image: \"%s\"\n", *input_img_path);
    printf("- Output Basename: \"%s\"\n", *out_path);
    printf("- Gaussian Radius: \"%.2f\"\n", *radius);
    printf("- Gaussian Standard Deviation Range: \"[%.2f,%.2f]\"\n", *stdev_start,*stdev_end);
    printf("- Gaussian Standard Deviation step: \"%.2f\"\n", *stdev_step);
    printf("- Alpha, Beta and Gamma: \"%.2f,%.2f,%.2f\"\n", *alpha,*beta,*gamma);
    puts("-----------------------");
}