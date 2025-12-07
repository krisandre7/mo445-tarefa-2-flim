
#include "ift.h"

typedef struct _colortable {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
} ColorTable;

int main(int argc, char **argv)
{
  Image  *orig,*img1,*img2;
  CImage *cimg2;
  AdjRel *A=NULL;
  ColorTable *color;
  int i;
  unsigned int iseed;

  /*------- -------------------------------------------------*/
#ifndef _WIN32
  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif

  /*--------------------------------------------------------*/
  
  if (argc != 3) {
    fprintf(stderr,"usage: labelcomp <orig.pgm> <dest.ppm>\n");
    exit(-1);
  }

  color = (ColorTable *) calloc(256,sizeof(ColorTable));
  iseed = (unsigned int)time(NULL);
  srand(iseed);


  for (i=0; i < 256; i++) {
    color[i].red   = (rand()%256);
    color[i].green = (rand()%256);
    color[i].blue  = (rand()%256);
    printf("%d %d %d\n",color[i].red,color[i].green,color[i].blue);
  }

  orig  = ReadImage(argv[1]);
  img1  = Threshold(orig,1,255);

  // Change adjacency to label letters with box 3x7, words with box 
  // 10x7, and lines with cross 30x7;

  A     = Cross(30,7);

  img2  = LabelBinComp(img1,A);
  printf("Maximum %d\n",MaximumValue(img2));
  cimg2 = CreateCImage(img2->ncols,img2->nrows);
  for (i=0; i < (img2->ncols*img2->nrows); i++) {
    if (img2->val[i]>0){
      cimg2->C[0]->val[i] = color[img2->val[i]].red;
      cimg2->C[1]->val[i] = color[img2->val[i]].green;
      cimg2->C[2]->val[i] = color[img2->val[i]].blue;
    }else{
      cimg2->C[0]->val[i] = 200;
      cimg2->C[1]->val[i] = 200;
      cimg2->C[2]->val[i] = 200;
    }
  }
  WriteCImage(cimg2,argv[2]);

  DestroyImage(&img1);
  DestroyCImage(&cimg2);
  DestroyImage(&orig);
  DestroyImage(&img2);
  DestroyAdjRel(&A);
  free(color);

  /* ---------------------------------------------------------- */

#ifndef _WIN32
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   
#endif

  return(0);
}
