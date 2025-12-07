#include "ift.h"


// void iftFLIMArtificial(iftFLIMArch * arch, char * param_bias_dir, char * param_normal_dir){


//     //------------------------------------------------------------------
//     //Create artificial images
//     iftMImage *input = iftCreateMImage(300,300, 1, 3);



//     for(int p=0;p<input->n; p++){
//         for(int b=0; b<input->m; b++){
//             input->val[p][b] = iftRandomNormalDouble()*0.7;
//         }
//     }

//     int atrous_factor = 1;
    
//     iftVoxel u;
//     u.x = 10;
//     u.y = 10;
//     u.z = 1;

//     //-----------------------------------------------------------------

//     float minv = iftMMinimumValue(input,-1);
//     float maxv = iftMMaximumValue(input,-1);


//     iftMImage *input2 = iftCopyMImage(input);

//     iftMImage **output_bias = NULL;

//     iftMImage **output_normal = NULL;

//     int l=0;
//     setenv("USE_BIAS", "1", 1);
//     output_bias = FLIMConvolutionalLayer(&input, 1, NULL, arch, l, l+1, atrous_factor, param_bias_dir);
//     unsetenv("USE_BIAS");
//     output_normal = FLIMConvolutionalLayer(&input2, 1, NULL, arch, l, l+1, atrous_factor, param_normal_dir);


//     float minv2 = iftMMinimumValue(output_normal[0],-1);
//     float minv1 = iftMMinimumValue(output_bias[0],-1);
    
//     printf("\n %f %f %f %f \n", minv,maxv, minv1, minv2);

//     float diff =0;
//     for(int p=0;p<output_bias[0]->n; p++){
//         for(int b=0; b<output_bias[0]->m; b++){
//             diff += abs(output_bias[0]->val[p][b]-output_normal[0]->val[p][b]);
//         }
//     }


//     u.x=0;
//     u.y=2;
//     u.z=1;

//     int q = iftMGetVoxelIndex(output_bias[0], u);

//     printf("----------------------------Diff------------------------------------\n");
//     for(int ch=0; ch<output_bias[0]->m; ch++){
//         printf("%.2f ", output_normal[0]->val[q][ch]);
//     }
//     printf("\n");

//     for(int ch=0; ch<output_bias[0]->m; ch++){
//         printf("%.2f ", output_bias[0]->val[q][ch]);
//     }
//     printf("\n");
//     printf("--------------------------------------------------------------------\n");

//     printf("\nDiff %f\n", diff);


//     iftMImage *diffimg = iftCopyMImage(output_bias[0]);


//     for (int p=0; p<output_bias[0]->n; p++){
//         for(int ch=0; ch<output_bias[0]->m; ch++){
//             diffimg->val[p][ch] = output_bias[0]->val[p][ch]-output_normal[0]->val[p][ch];

//             if(diffimg->val[p][ch] > 0.01){
//                 diffimg->val[p][ch] = 1;
//             }
//             else{
//                 diffimg->val[p][ch] = 0;
//             }

//         }
//     }

//     iftWriteMImage(diffimg, "diffimg.mimg");

//     iftDestroyMImage(&output_bias[0]);
//     iftDestroyMImage(&output_normal[0]);
//     iftFree(output_bias);
//     iftFree(output_normal);
// }

iftMImage *ReadInputMImage2(char *filename) {
    iftImage  *img     = iftReadImageByExt(filename);
    iftMImage *input   = iftImageToMImage(img, LABNorm2_CSPACE); 
    iftDestroyImage(&img);    
    return (input);
}

void iftFLIMCompareBias( iftFLIMArch *arch,  char * image_dir,  char * seeds_dir,
                         const char * output_dir){

    
    iftMakeDir(output_dir);

    
    char filename[200];
    sprintf(filename, "%s/%s", output_dir, "param_flim_bias");


    setenv("USE_BIAS", "1", 1);
    //learn model with normal
    iftMakeDir(filename);
    iftFLIMLearnModelPCA(image_dir,seeds_dir, filename, arch);

    //learn model bias
    unsetenv("USE_BIAS");
    sprintf(filename, "%s/%s", output_dir, "param_flim");
    iftMakeDir(filename);
    iftFLIMLearnModelPCA(image_dir,seeds_dir, filename, arch);

    //check if learned models are equivalent
    //iftVerifyBiasedFLIM()


    //create fake image
    // iftMImage *aux = iftCreateMImage(300,300, 1, 3);

    // for(int p=0;p<aux->n; p++){
    //     for(int b=0; b<aux->m; b++){
    //         aux->val[p][b] = iftRandomNormalFloat(0, 0.1);
    //     }
    // }
    // iftMImage *input = iftNormalizeByBand(aux);

    // printf("min %f max %f", iftMMinimumValue(input, -1), iftMMaximumValue(input,-1));

    // iftDestroyMImage(&aux);

    //read image
    iftMImage *input = ReadInputMImage2("orig/000001_00000001.png");

    //save fake image
    sprintf(filename, "%s/%s", output_dir, "out");
    iftMakeDir(filename);
    sprintf(filename, "%s/%s/%s", output_dir, "out", "tmp.mimg");
    iftWriteMImage(input,filename);


    //create fake csv image
    FILE *fp;
    char file_list[100];
    sprintf(file_list, "%s/%s", output_dir, "files.csv");
    fp = fopen(file_list, "w");
    fprintf(fp,"tmp.mimg\n");

    fclose(fp);

    char param_dir[200], feat_dir[200];

    //extract features with bias
    setenv("USE_BIAS", "1", 1);
    sprintf(filename, "%s/%s", output_dir, "out");
    sprintf(param_dir, "%s/%s", output_dir, "param_flim_bias");
    sprintf(feat_dir, "%s/%s", output_dir, "feat_bias");
    iftMakeDir(feat_dir);
    iftFLIMExtractFeatures(filename, file_list, arch, param_dir, feat_dir, NULL, 0);

    //extract features without bias
    unsetenv("USE_BIAS");
    sprintf(filename, "%s/%s", output_dir, "out");
    sprintf(param_dir, "%s/%s", output_dir, "param_flim");
    sprintf(feat_dir, "%s/%s", output_dir, "feat");
    iftMakeDir(feat_dir);
    iftFLIMExtractFeatures(filename, file_list, arch, param_dir, feat_dir, NULL, 0);


    //comparing image
    sprintf(filename, "%s/%s/%s", output_dir, "feat_bias", "tmp.mimg");
    iftMImage *img_bias = iftReadMImage(filename);


    sprintf(filename, "%s/%s/%s", output_dir, "feat", "tmp.mimg");
    iftMImage *img = iftReadMImage(filename);

    

    double diff = 0;


    for(int p=0;p<img->n; p++){
        for(int b=0; b<img->m; b++){
            diff += fabs(img->val[p][b]-img_bias->val[p][b]);
        }
    }


    printf("-------------------------------------------------------------\n\n");
    printf("Diff between FLIM and Biased-FLIM is:  %lf\n\n", diff);
    printf("-------------------------------------------------------------\n");

    iftDestroyMImage(&img_bias);
    iftDestroyMImage(&img);
    iftDestroyMImage(&input);
    //iftRemoveDir(output_dir);

}


int main(int argc, char *argv[])
{
    timer *tstart;

    if ((argc != 4) && (argc!=5))
      iftError("Usage: iftFLIM-ExtractFeatures P1 P2 P3\n"
	       "P1: input  FLIM network architecture (.json)\n"
	       "P2: input  folder with original images (.png, .nii.gz)\n"
           "P3: input  folder with the seeds\n"
           "P4 (optional): dir to save tmp files",
	       "main");
    
    tstart = iftTic();

    iftFLIMArch *arch      = iftReadFLIMArch(argv[1]);
    char *input_images_dir        = argv[2];
    char *seeds_dir         = argv[3];
    
    char *output_dir= NULL;

    if (argc==5){
        output_dir=argv[4];
    }else{
        output_dir = (char*) malloc(9*sizeof(char));
        strcpy(output_dir, "tmp_bias");
    }

    
    iftFLIMCompareBias(arch, input_images_dir, seeds_dir, output_dir);
    
    iftDestroyFLIMArch(&arch);

    if (argc!=5)
        free(output_dir);
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}


