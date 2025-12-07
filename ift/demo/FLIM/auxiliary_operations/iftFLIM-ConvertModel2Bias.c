#include "ift.h"


void verify_arguments(iftFLIMArch * arch, char * input_dir){

    if(!iftDirExists(input_dir)){
        iftError("Input dir does not exist!!", "vefiry_arguments");
    }

    char filename[200];
    for(int i=0; i<arch->nlayers; i++){
        sprintf(filename, "%s/conv%d-kernels.npy",input_dir, i+1);
        if(!iftFileExists(filename)){
            sprintf(filename, "conv%d-kernels.npy does not exist!!!", i+1);
            iftError(filename, "verify_aguments");
        }

        sprintf(filename, "%s/conv%d-mean.txt",input_dir, i+1);
        if(!iftFileExists(filename)){
            sprintf(filename, "conv%d-mean.txt does not exist!!!", i+1);
            iftError(filename, "verify_aguments");
        }

        sprintf(filename, "%s/conv%d-stdev.txt",input_dir, i+1);
        if(!iftFileExists(filename)){
            sprintf(filename, "conv%d-stdev.txt does not exist!!!", i+1);
            iftError(filename, "verify_aguments");
        }
    }
}

int main(int argc, char *argv[]){


    if (argc != 5){
        iftError("iftFLIM_Convert_Model2bias P1 P2 P3\n"
        "[P1] architecture json file\n"
        "[P2] input model with parameters\n"
        "[P3] output dir\n"
        "[P4] Input channels of the first layer\n",
        "main");
    }

    char * param_dir  = argv[2];
    char * output_dir = argv[3];
    int ninput_channels      = atoi(argv[4]);

    //load architecture
    iftFLIMArch *arch      = iftReadFLIMArch(argv[1]);

    //check arguments
    verify_arguments(arch, param_dir);

    iftFLIMConvertModel2BIAS(arch, ninput_channels, param_dir, output_dir);

    iftDestroyFLIMArch(&arch);

    return 0;
}