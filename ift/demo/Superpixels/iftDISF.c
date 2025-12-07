#include "ift.h"


int main(int argc, char *argv[])
{
    timer *tstart;

    if ((argc!=4)&&(argc!=6))
        iftError("Usage: iftDISF <...>\n"
		 "[1] input multiband image with the activations of the last convolutional layer \n"
		 "[2] input number of supervoxels \n"
		 "[3] output label image of supervoxels (.nii.gz, .scn, .pgm, .png) \n"
		 "[4] Optional: label image (.nii.gz, .scn, .pgm, .png) \n"
		 "[5]           object label (-1 for all, 1, 2, etc.) \n",		 
		 "main");

    tstart = iftTic();

    iftMImage     *img = iftReadMImage(argv[1]);
    int       nsupervoxels = atoi(argv[2]);
    iftImage  *mask, *label=NULL;
    iftAdjRel *A;

    /* Compute ISF supervoxels */
    if (iftIs3DMImage(img)){
      A      = iftSpheric(1.0);
    } else {
      A      = iftCircular(1.0);
    }

    if (argc == 4) 
      mask   = iftSelectImageDomain(img->xsize,img->ysize,img->zsize);
    else {
      label = iftReadImageByExt(argv[4]);
      int object = atoi(argv[5]);

      if (object == -1)
	mask = iftThreshold(label,1,IFT_INFINITY_INT,1);
      else
	mask = iftThreshold(label,object,object,1);

      iftDestroyImage(&label);
    }

    if ((img->xsize != mask->xsize)||(img->ysize != mask->ysize)||(img->zsize != mask->zsize))
      iftError("Input and label images must have the same domain","iftDISF.c");
    
    int numinitseeds = 20 * nsupervoxels;

    if (iftIs3DMImage(img)){
      int maxnumofseeds = iftRound(pow(iftMin(iftMin(img->xsize,img->ysize),img->zsize)/3.0,3));
      numinitseeds = iftMin(numinitseeds,maxnumofseeds);
    }else{ 
      int maxnumofseeds = iftRound(pow(iftMin(img->xsize,img->ysize)/3.0,2));
      numinitseeds = iftMin(numinitseeds,maxnumofseeds);
    }
    
    if (mask != NULL) {
      int masksize=0;
      for (int p=0; p < mask->n; p++)
	if (mask->val[p])
	  masksize++;
      
      numinitseeds = iftMin(masksize/3.0,numinitseeds); 
    }

    numinitseeds = iftMax(numinitseeds, nsupervoxels);

    printf("Initial number of seeds %d\n",numinitseeds);
    
    label   = iftDISF(img, A, numinitseeds, nsupervoxels, mask);
    iftDestroyImage(&mask);
    iftDestroyMImage(&img);
    iftDestroyAdjRel(&A);
  
    iftWriteImageByExt(label,argv[3]);
    iftDestroyImage(&label);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
