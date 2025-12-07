#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage       *orig=NULL,*surf_skel=NULL;
  iftFImage      *geo_skel=NULL;
  iftImage       *msskel=NULL;
  float           thres;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/


  if (argc<2)
    iftError("Usage: iftSkel <binary.[vtk,scn]>","main");

  if (iftEndsWith(argv[1],"vtk")) 
      orig   = iftReadVTKImage(argv[1]);
  else
      orig   = iftReadImage(argv[1]);

  /* Count the number of connected components */

  int             number_of_components;
  int nvoxels    = 0;
  iftAdjRel *A   = iftSpheric(1.0);
  iftSet *B      = iftObjectBorderSet(orig,A);
  iftDestroyAdjRel(&A);
  A              = iftSpheric(sqrtf(3.0));
  iftImage *aux  = iftFastLabelComp(orig,A);
  //  iftWriteImage(aux,"components.scn");
  number_of_components = iftMaximumValue(aux);
  printf("number of components %d\n",number_of_components);
  iftDestroyAdjRel(&A);
  iftDestroyImage(&aux);
  while (B != NULL)
  {
    nvoxels++;
    iftRemoveSet(&B);
  }
  printf("Number of surface points %d\n",nvoxels);

  /* Multiscale skeletonization starts here */
 
  printf("Computing multiscale skeleton by geodesic length\n");

  t1         = iftTic();
  geo_skel   = iftMSSkel(orig);
  t2         = iftToc();
  fprintf(stdout,"MS skeletons by geodesic length in %f ms\n",iftCompTime(t1,t2));

  iftDestroyImage(&orig);

  iftFWriteVTKImage(geo_skel,"multiscale-skeletons.vtk");
  msskel = iftFImageToImage(geo_skel,4095);
  iftWriteImage(msskel,"multiscale-skeletons.scn");
  iftDestroyImage(&msskel);

  printf("Type a number between (0,1] to threshold skeleton or 0 to exit\n");
  scanf("%f",&thres);

  float maxval=iftFMaximumValue(geo_skel);

  while (thres != 0) {

    t1     = iftTic();
    surf_skel  = iftSurfaceSkeleton(geo_skel,thres*maxval,number_of_components);
    t2     = iftToc();
    fprintf(stdout,"Skeletonization in %f ms\n",iftCompTime(t1,t2));
    iftWriteImage(surf_skel,"skeleton.scn");
    iftWriteVTKImage(surf_skel,"skeleton.vtk");
    iftDestroyImage(&surf_skel);  

    printf("Type a number between (0,1] to threshold skeleton or 0 to exit\n");
    scanf("%f",&thres);
  }

  iftDestroyFImage(&geo_skel);

  /* ---------------------------------------------------------- */


  return(0);
}



