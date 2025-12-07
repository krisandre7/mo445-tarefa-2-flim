#include "ift.h"

void iftNormalizeBandsInPlace(iftMImage *mimg)
{
  for (int b=0; b < mimg->m; b++) {
    float min = IFT_INFINITY_FLT;
    float max = IFT_INFINITY_FLT_NEG;
    for (int p = 0; p < mimg->n; p++) {
      if (mimg->val[p][b] < min)
	min = mimg->val[p][b];
      if (mimg->val[p][b] > max)
	max = mimg->val[p][b];
    }
    if (!iftAlmostZero(max-min))
      for (int p = 0; p < mimg->n; p++) {
	mimg->val[p][b] = (mimg->val[p][b]-min)/(max-min);
      }
  }
}

int main(int argc, char *argv[])
{
    if (argc!=4)
        iftError("Usage: iftSkipConnection <...>\n"
	       "[1] input_layer_1: Directory containing the input layer 1\n"
	       "[2] input_layer_2: Directory containing the input layer 2\n"
	       "[3] output_dir: Directory where the resulting mimages will be saved\n",
	       "main");

    timer *tstart = iftTic();

    iftFileSet * fs1 = iftLoadFileSetFromDirOrCSV(argv[1], 1, true);
    iftFileSet * fs2 = iftLoadFileSetFromDirOrCSV(argv[2], 1, true);
    iftMakeDir(argv[3]);
    
    if(fs1->n != fs2->n)
        iftError("The number of images in the two input layers must be the same", "iftSkipConnection.c");

    for(int i = 0; i < fs1->n; i++) {
        printf("Processing image %d of %ld\r", i + 1, fs1->n);
        char *basename1 = iftFilename(fs1->files[i]->path,".mimg");
        char *basename2 = iftFilename(fs2->files[i]->path,".mimg");

        if(!iftCompareStrings(basename1, basename2))
            iftError("The set of images in the two input layers must be the same", "iftSkipConnection.c");

        iftMImage *mimg1 = iftReadMImage(fs1->files[i]->path);
	//	iftNormalizeBandsInPlace(mimg1);
        iftMImage *mimg2 = iftReadMImage(fs2->files[i]->path);
	//	iftNormalizeBandsInPlace(mimg2);

        if(mimg1->n != mimg2->n) {
            iftMImage *aux = iftResizeMImage(mimg1, mimg2->xsize, mimg2->ysize, mimg2->zsize);
            iftDestroyMImage(&mimg1);
            mimg1 = iftCopyMImage(aux);
            iftDestroyMImage(&aux);
        }

        iftMImage *mimg3 = iftCreateMImage(mimg2->xsize, mimg2->ysize, mimg2->zsize, mimg1->m + mimg2->m);
	
        #pragma omp parallel for
        for(int p = 0; p < mimg1->n; p++) {
            int band = 0;
            for(int b = 0; b < mimg1->m; b++) {
                mimg3->val[p][band] = mimg1->val[p][b];
                band++;
            }
            for(int b = 0; b < mimg2->m; b++) {
                mimg3->val[p][band] = mimg2->val[p][b];
                band++;
            }
        }
	
        char filename[2048];
        sprintf(filename,"%s/%s.mimg",argv[3],basename1);
        iftWriteMImage(mimg3, filename);
        iftDestroyMImage(&mimg1);
        iftDestroyMImage(&mimg2);
        iftDestroyMImage(&mimg3);
    }

    printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
