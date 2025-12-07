#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage  *img,*rend;
  iftFImage *scene;
  iftGraphicalContext *gc;
  timer     *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/
  size_t MemDinInicial, MemDinFinal;
  MemDinInicial = iftMemoryUsed();

  /*--------------------------------------------------------*/

  if ((argc!=8)&&(argc!=7))
      iftError("Usage: iftVolumeRender <image.scn> <tilt> <spin> <min_val> <max_val> <grad_thres (optional) <max_opac [0-1]> ", "main");

  img    = iftReadImage(argv[1]);
  iftAdjRel *A=iftSpheric(3.0);
  iftImage *grad = iftImageGradientMagnitude(img,A);
  iftWriteImage(grad,"grad.scn");
  iftDestroyAdjRel(&A);
  
  scene  = iftImageToFImage(img);
  gc     = iftCreateGraphicalContext(scene,NULL);
  if (argc==7)
    iftSetSceneOpacity(gc,atof(argv[4]),atof(argv[5]),grad,IFT_NIL,atof(argv[6]));
  else
    iftSetSceneOpacity(gc,atof(argv[4]),atof(argv[5]),grad,atoi(argv[6]),atof(argv[7]));
    
  iftDestroyImage(&img);
  iftDestroyImage(&grad);
  iftSetViewDir(gc,atof(argv[2]),atof(argv[3]));

  t1 = iftTic(); 
  
  rend   = iftVolumeRender(gc);

  t2     = iftToc();

  fprintf(stdout,"Rendering in %f ms\n",iftCompTime(t1,t2));
  
  img = iftNormalize(rend,0,255);
  iftWriteImageByExt(img,"result.png");
  iftDestroyImage(&rend);
  iftDestroyImage(&img);
  iftDestroyFImage(&scene);
  iftDestroyGraphicalContext(gc);


  /* ---------------------------------------------------------- */

  MemDinFinal = iftMemoryUsed();
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%ld, %ld)\n",
           MemDinInicial,MemDinFinal);

  return(0);
}

