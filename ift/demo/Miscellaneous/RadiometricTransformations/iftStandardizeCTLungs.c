#include "ift.h"

#define EXT3D ".nii.gz"
#define DX    20    /* controls the size of the plateau for maxima detection */
#define I0    1000  /* maps the first maximum to intensity I0 */
#define I1    3000  /* maps the second maximum to intensity I1 */
#define I2    4095  /* maps voxels above the second maximum to intensity I2 */


int THRES=200;   /* ignores plateaus with altitude less than or equal to a threshold */

int *LandMarks(float *hist, int n)
{
  int   *xmax = iftAllocIntArray(n);
  int    dx   = DX;
  
  /* Detect the maxima */
  
  for (int x=dx; x < n-dx; x++){
    if (hist[x]>THRES){
      char max=1;
      for (int delta=-dx; delta <= dx; delta++)
          if (hist[x] < hist[x+delta]){
            max=0;
            break;
          }
      if (max)
          xmax[x]=1;
    }
  }
  
  /* Return the highest and last maxima */
  
  int *landmark = iftAllocIntArray(2);
  float max = 0;
  for (int x=0; x < n; x++){
    if (xmax[x]>0){
      if (hist[x] > max){
        max = hist[x];
        landmark[0]=x;
      }
    }
  }

  for (int x=n-1; x >=0; x--){
    if (xmax[x]>0){
      landmark[1]=x;
      break;
    }
  }
  iftFree(xmax);
  
  return(landmark);
}


int main(int argc, char *argv[])
{
  timer *tstart;
  
  if ((argc < 4) || (argc > 5))
    iftError("Usage: iftStandardizeCTLungs <...>\n"
         "[1] Input dir with CT images. \n"
         "[2] input dir with label image\n"
         "[3] OPTIONAL Plateu threshold (default 200)\n"
         "[4] Output dir\n",           
         "main");
  
  tstart = iftTic();

  char *out_path = argv[3];
  if (argc == 5){
    THRES = atoi(argv[3]);
    out_path = argv[4];
  }
  
  iftFileSet *fs_orig    = iftLoadFileSetFromDirBySuffix(argv[1],EXT3D, 1);
  iftFileSet *fs_label   = iftLoadFileSetFromDirBySuffix(argv[2],EXT3D, 1);
  if (fs_orig->n != fs_label->n)
    iftError("Original and label images are not the same in number","main");    
  int nimages            = fs_orig->n;   
  iftMakeDir(out_path);
  char filename[200];
  
  for (int i=0; i < nimages; i++){
    char *basename       = iftFilename(fs_orig->files[i]->path,EXT3D);
    
    printf("Processing file %s: %d of %d files\r", basename, i + 1, nimages);
    fflush(stdout);
    
    sprintf(filename,"%s/%s%s",argv[1],basename,EXT3D);
    iftImage *input       = iftReadImageByExt(filename);
    sprintf(filename,"%s/%s%s",argv[2],basename,EXT3D);
    iftImage *label       = iftReadImageByExt(filename);
    
    float hist[I2+1];
    for (int i=0; i < I2+1; i++){
      hist[i]=0;
    }
    
    int Imin = IFT_INFINITY_INT;
    for (int p=0; p < input->n; p++){
      if ((label->val[p] != 0)&&(input->val[p]>0)){
        if (input->val[p]<Imin)
          Imin = input->val[p];       
        hist[input->val[p]]++;
      }
    }

    /* Find landmarks on the histogram and standardize the histogram
       according to those landmarks */
    
    int *landmark = LandMarks(hist,I2+1); 
    printf("%d\n",landmark[0]);
    printf("%d\n",landmark[1]);

    for (int p=0; p < input->n; p++){
      if (label->val[p]>0){
        if (input->val[p] <= landmark[0])
          input->val[p] = I0*((float)iftMax(input->val[p]-Imin,0)/((float)landmark[0]-Imin));
        else{
          if (input->val[p] <= landmark[1])
            input->val[p] = I1*((float)input->val[p]-(float)landmark[0])/((float)landmark[1]-(float)landmark[0]) + I0;
          else
            input->val[p] = I2;
        }
      }
    }
    iftFree(landmark);    
    sprintf(filename,"%s/%s%s",out_path,basename,EXT3D);
    iftWriteImageByExt(input,filename);     
    iftDestroyImage(&input);
    iftDestroyImage(&label);
    iftFree(basename);
  }

  iftDestroyFileSet(&fs_orig);
  iftDestroyFileSet(&fs_label);

  printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  return (0);
}

