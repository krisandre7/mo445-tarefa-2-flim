//
// Created by azaelmsousa on 23/06/21.
//

#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc!=4)
        iftError("Usage: iftSelectKernelsManual P1 P2 P3\n"
                 "[1] input kernel bank (.npy)\n"
                 "[2] input a file with selected kernels selected_samples are kernels starting from 0 (.json)\n"
                 "[3] output output bank with selected kernels (.npy)\n",
                 "main");

    tstart = iftTic();

    char *kernel_bank_path       = argv[1];
    char *selected_kernels_path  = argv[2];
    char *output_kernel_bank     = argv[3];

    iftMatrix *output_kernels = iftFLIMSelectKernelsManual(kernel_bank_path,selected_kernels_path);
    iftWriteMatrix(output_kernels,output_kernel_bank);
    iftDestroyMatrix(&output_kernels);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
