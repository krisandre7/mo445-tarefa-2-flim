#include "ift.h"


void SaveResults(char *dir, char *basename, iftMImage *mimg)
{
    char filename[200];

    sprintf(filename, "%s/%s.mimg", dir, basename);
    iftWriteMImage(mimg, filename);
}

int main(int argc, char *argv[])
{
    timer *tstart = NULL;

    if ((argc != 3)&&(argc != 4))
        iftError("Usage: iftConvertImagesToMImages <...>\n"
                 "[1] input_images: Input directory/csv file with original images\n"
                 "[2] input_labels: Input directory/csv file with label images (optional)\n"
                 "[3] output_directory: Output directory with mimages\n",
                 "main");

    tstart = iftTic();

    iftFileSet *fs = iftLoadFileSetFromDirOrCSV(argv[1], 0, 1);
    char ext[10];
    sprintf(ext,"%s",iftFileExt(fs->files[0]->path));
    
    if (argc == 4){
      iftMakeDir(argv[3]);
      int first = 0;
      int last  = fs->n - 1;

      char filename[200];
      
      for (int i = first; i <= last; i++) {
        char *basename = iftFilename(fs->files[i]->path, ext);
        printf("Processing file %s: %d of %ld files\r", basename, i + 1, fs->n);
        fflush(stdout);
        iftImage *orig  = iftReadImageByExt(fs->files[i]->path);
	sprintf(filename,"%s/%s%s",argv[2],basename,ext);
        iftImage *label = iftReadImageByExt(filename);
	
        iftImage *obj   = iftMask(orig,label);
      	iftMImage *mimg = NULL;

	mimg = iftImageToMImage(obj, LABNorm2_CSPACE); 

	SaveResults(argv[3], basename, mimg);
        iftDestroyImage(&orig);
        iftDestroyImage(&label);
	iftDestroyImage(&obj);
        iftDestroyMImage(&mimg);
	iftFree(basename);
      }
    } else {
      iftMakeDir(argv[2]);
      int first = 0;
      int last  = fs->n - 1;
      for (int i = first; i <= last; i++) {
        char *basename = iftFilename(fs->files[i]->path, ext);
        printf("Processing file %s: %d of %ld files\r", basename, i + 1, fs->n);
        fflush(stdout);
        iftImage *orig  = iftReadImageByExt(fs->files[i]->path);
        iftMImage *mimg = NULL;
	mimg = iftImageToMImage(orig, LABNorm2_CSPACE); 
        SaveResults(argv[2], basename, mimg);
        iftDestroyImage(&orig);
        iftDestroyMImage(&mimg);
      }
    }
    printf("\n");

    iftDestroyFileSet(&fs);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
