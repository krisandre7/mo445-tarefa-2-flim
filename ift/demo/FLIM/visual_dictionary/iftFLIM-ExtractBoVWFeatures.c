#include "ift.h"
#include "iftFLIM-BoVW.h"

void MyFLIMExtractFeaturesPerImage(iftFileSet *fs_images, int first, int last, char *orig_dir, iftFLIMArch *arch, char *param_dir, char *feat_dir) {
  
    iftMImage **input = NULL, **output = NULL;
    char filename[200];

    printf("\n");

    for (int i = first; i <= last; i++) {
        input = (iftMImage **) calloc(1, sizeof(iftMImage *));     
	char ext[20];
	sprintf(ext, "%s", iftFileExt(fs_images->files[i]->path));
	sprintf(filename, "%s/%s", orig_dir, fs_images->files[i]->path);
	if (strcmp(ext,".mimg") == 0){
	  input[0] = iftReadMImage(filename);
	}else{
	  input[0] = MyReadInputMImage(filename);
        }
        char *basename = iftFilename(fs_images->files[i]->path, ext);
  
        printf("Processing file %s: image %d of %d files\n", basename, i + 1, last - first + 1);
        fflush(stdout);

        for (int l = 0; l < arch->nlayers; l++) {
	  output = MyConvolutionalLayer(input, 1, arch, l, l+1, 1, param_dir);
	  iftDestroyMImage(&input[0]);
	  input[0] = output[0];
	  output[0] = NULL;
	  iftFree(output);
        }

        sprintf(filename, "%s/%s.mimg", feat_dir, basename);
        iftFree(basename);
        iftWriteMImage(input[0], filename);
        iftDestroyMImage(&input[0]);
        iftFree(input);
    }
}

void iftFLIMExtractNewFeatures(char *orig_dir, char *image_list, iftFLIMArch *arch, char *param_dir, char *feat_dir) {
    iftFileSet *fs_images = iftLoadFileSetFromDirOrCSV(image_list, 1, 1);
    int nimages = fs_images->n;
    int first = 0, last = nimages - 1;
    
    MyFLIMExtractFeaturesPerImage(fs_images, first, last, orig_dir, arch, param_dir, feat_dir);
  
  iftDestroyFileSet(&fs_images);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 6)
      iftError("Usage: iftFLIM-ExtractNewFeatures P1 P2 P3 P4 P5 P6\n"
	       "P1: input  FLIM network architecture (.json)\n"
	       "P2: input  folder with the original images (.png, .nii.gz)\n"
	       "P3: input  list of images for feature extraction (.csv)\n"
	       "P4: input  folder with the FLIM network parameters\n"
	       "P5: output folder with the resulting image features\n",
	       "main");
    
    tstart = iftTic();

    iftFLIMArch *arch      = iftReadFLIMArch(argv[1]);
    char *orig_dir         = argv[2];
    char *image_list       = argv[3];
    char *param_dir        = argv[4];
    char *feat_dir         = argv[5];

    iftMakeDir(feat_dir);

    /* Extract image features using the FLIM architecture and
       parameters, and save the results in the output folder */
    
    iftFLIMExtractNewFeatures(orig_dir, image_list, arch, param_dir, feat_dir);
    iftDestroyFLIMArch(&arch);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
