//
// Created by azaelmsousa on 23/06/21.
//

#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 7)
        iftError("Usage: iftFLIM-LearnLayer P1 P2 P3 P4 P5 P6\n"
                 "P1: input  FLIM network architecture (.json)\n"
                 "P2: input  folder with the activation or original images (.nii.gz, .png, .mimg)\n"
                 "P3: input  index of layer to be trained ([1,nlayers])\n"
                 "P4: input  folder with the training markers (-seeds.txt)\n"
                 "P5: output folder with the FLIM network parameters per layer\n"
                 "P6: output folder with FLIM activations\n",
                 "main");

    tstart = iftTic();

    iftFLIMArch *arch      = iftReadFLIMArch(argv[1]);
    char *orig_dir         = argv[2];
    int layer_index        = atoi(argv[3]);
    char *markers_dir      = argv[4];
    char *param_dir        = argv[5];
    char *out_dir          = argv[6];

    iftMakeDir(param_dir);
    iftMakeDir(out_dir);

    /* Learn the FLIM feature extraction model and save parameters into the output folder */

    iftFLIMLearnLayer(orig_dir,markers_dir, param_dir, layer_index, arch, out_dir);
    iftDestroyFLIMArch(&arch);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
