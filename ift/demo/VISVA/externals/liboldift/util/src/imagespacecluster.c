#include "ift.h"

Image *RemoveRegionByArea(Image *label, int area)
{
  Curve *label_area;
  int p,n;
  AdjRel *A;
  Image *rlabel,*nlabel;

  A          = Circular(1.5);
  rlabel     = LabelComp(label,A,0);
  label_area = Histogram(rlabel);
  n          = label->ncols*label->nrows;
  for (p=0; p < n; p++){
    if (label_area->Y[rlabel->val[p]] < area)
      rlabel->val[p]=0; // force basins in regions to be removed
  }
  DestroyCurve(&label_area);
  nlabel = FastAreaClose(rlabel,area); // remove regions 
  DestroyImage(&rlabel);
  rlabel     = LabelComp(nlabel,A,0); // relabel final image
  DestroyAdjRel(&A);
  DestroyImage(&nlabel);
  return(rlabel);
}

int main(int argc, char **argv)
{
  timer *t1=NULL,*t2=NULL;
  Image    *img=NULL,*label=NULL,*final_label=NULL;
  Subgraph *sg=NULL;
  Features *f=NULL;
  CImage *cimg2=NULL,*cimg1=NULL;
  char ext[10],*pos;
  float di,df;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=6)
    Error("Usage must be: imagecluster <image> <kmax> <di> <volume> <area>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);

  t1 = Tic();

  if (strcmp(ext,"pgm")==0){
    img = ReadImage(argv[1]);
    sg = UnifSampl(img,NULL,16,16,0,0);
    f  = LMSImageFeats(img,5);
  }else{
    if (strcmp(ext,"ppm")==0){
      cimg1 = ReadCImage(argv[1]);
      cimg2 = CImageRGBtoLab(cimg1);
      sg = UnifSampl(cimg1->C[0],NULL,16,16,0,0);
      f=LMSCImageFeats(cimg2,3);
      DestroyCImage(&cimg2);
      img = cimg1->C[1];
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  SetSubgraphFeatures(sg,f);
  opf_BestkMinCut(sg,1, atoi(argv[2]));
  df = sg->df;
  di = atof(argv[3]);

  DestroySubgraph(&sg);

  sg=UnifSampl(img,NULL,1,1,0,0);
  SetSubgraphFeatures(sg,f);
  DestroyFeatures(&f);
  sg->df=df;
  sg->di=di;
  label = ImageLabel(sg,img,atoi(argv[4]));
  final_label=RemoveRegionByArea(label,atoi(argv[5]));
  t2 = Toc();
  cimg2     = DrawLabeledRegions(img,final_label);
  WriteCImage(cimg2,"result.ppm");
  DestroySubgraph(&sg);
  if (strcmp(ext,"pgm")==0)
    DestroyImage(&img);
  else
    DestroyCImage(&cimg1);
  DestroyImage(&label);
  DestroyImage(&final_label);
  DestroyCImage(&cimg2);

  fprintf(stdout,"imagescluster in %f ms\n",CTime(t1,t2));

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  return(0);
}
