#include "ift.h"

int main(int argc, char *argv[])
{
    timer *tstart = NULL;

    if (argc != 6)
        iftError("Usage: iftReformatImagesIntoSameDomain <P1> <P2> <P3> <P4> <P5>\n"
                 "P1: Folder with input images (.png, .nii.gz, .mimg, etc)\n"
                 "P2: Output xsize\n"
                 "P3: Output ysize\n"
                 "P4: Output zsize\n"
                 "P5: Folder with the output images\n",
                 "main");

    tstart = iftTic();

    iftFileSet *fs  = iftLoadFileSetFromDirOrCSV(argv[1], 1, true);
    const char *ext = iftFileExt(fs->files[0]->path);
    printf("%s\n",ext);
    int xsize       = atoi(argv[2]);
    int ysize       = atoi(argv[3]);
    int zsize       = atoi(argv[4]);
    if ((xsize<=0)||(ysize<=0)||(zsize<=0))
      iftError("Invalid domain %d x %d x %d","main",xsize,ysize,zsize);	
    char *out_dir   = argv[5];
    iftMakeDir(out_dir);
    char filename[200];

#pragma omp parallel for	
    for (int i = 0; i < fs->n; i++) {
      char *basename = iftFilename(fs->files[i]->path, ext);
      if (strcmp(ext,".mimg")==0){	
	iftMImage *input  = iftReadMImage(fs->files[i]->path);
	if (iftIs3DMImage(input)){
	  float sx = (float) xsize / (float) input->xsize;
	  float sy = (float) ysize / (float) input->ysize;
	  float sz = (float) zsize / (float) input->zsize;
	  iftMImage *output = iftMInterp(input,sx, sy, sz);
	  sprintf(filename,"%s/%s%s",out_dir,basename,ext);
	  iftWriteMImage(output,filename);
	  iftDestroyMImage(&input);
	  iftDestroyMImage(&output);
	} else {
	  float sx = (float) xsize / (float) input->xsize;
	  float sy = (float) ysize / (float) input->ysize;
	  iftMImage *output = iftMInterp2D(input,sx, sy);
	  sprintf(filename,"%s/%s%s",out_dir,basename,ext);
	  iftWriteMImage(output,filename);
	  iftDestroyMImage(&input);
	  iftDestroyMImage(&output);
	}
      } else {
	iftImage *input  = iftReadImageByExt(fs->files[i]->path);
	if (iftIs3DImage(input)){
	  float sx = (float) xsize / (float) input->xsize;
	  float sy = (float) ysize / (float) input->ysize;
	  float sz = (float) zsize / (float) input->zsize;
	  iftImage *output = iftInterp(input,sx, sy, sz);
	  sprintf(filename,"%s/%s%s",out_dir,basename,ext);
	  iftWriteImageByExt(output,filename);
	  iftDestroyImage(&input);
	  iftDestroyImage(&output);
	} else {
	  float sx = (float) xsize / (float) input->xsize;
	  float sy = (float) ysize / (float) input->ysize;
	  iftImage *output = iftInterp2D(input,sx, sy);
	  sprintf(filename,"%s/%s%s",out_dir,basename,ext);
	  iftWriteImageByExt(output,filename);
	  iftDestroyImage(&input);
	  iftDestroyImage(&output);	  
	}
      }      
      iftFree(basename);
    }
    iftDestroyFileSet(&fs);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
