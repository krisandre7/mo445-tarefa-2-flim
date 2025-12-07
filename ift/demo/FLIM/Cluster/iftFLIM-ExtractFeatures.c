#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart;

    if ((argc != 7)&&(argc != 8))
      iftError("Usage: iftFLIM-ExtractFeatures P1 P2 P3 P4 P5 P6 P7 (optional)\n"
	       "P1: input  FLIM network architecture (.json)\n"
	       "P2: input  folder with the original images (.png, .nii.gz)\n"
	       "P3: input  list of images for feature extraction (.csv)\n"
	       "P4: input  folder with the FLIM network parameters\n"
	       "P5: output folder with the resulting image features\n"
	       "P6: GPU device (0,1,..) or CPU when IFT_GPU=0\n"
	       "P7 (optional): input folder with object masks \n",
	       "main");
    
    tstart = iftTic();

    iftFLIMArch *arch      = iftReadFLIMArch(argv[1]);
    char *orig_dir         = argv[2];
    char *image_list       = argv[3];
    char *param_dir        = argv[4];
    char *feat_dir         = argv[5];
    int   device           = atoi(argv[6]);
    char *object_dir       = NULL;
    
    if (argc == 8)
      object_dir = argv[7];

    iftMakeDir(feat_dir);

    /* Extract image features using the FLIM architecture and
       parameters, and save the results in the output folder */
    
    iftFLIMExtractFeatures(orig_dir, image_list, arch, param_dir, feat_dir, object_dir, device);
    iftDestroyFLIMArch(&arch);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
