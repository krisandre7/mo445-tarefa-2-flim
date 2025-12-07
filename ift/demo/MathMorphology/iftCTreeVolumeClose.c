#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage       *img[2];
  timer          *t1=NULL,*t2=NULL;
  iftCompTree    *ctree;
  int             thres,opt;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=2)
    iftError("Usage: iftCTreeVolumeClose <image.[pgm,ppm,scn]>","main");

  img[0]   = iftReadImageByExt(argv[1]);    
  ctree    = iftCreateMinTree(img[0]);

  printf("Type 1 to continue and 0 to exit\n");
  scanf("%d",&opt);
  getchar();

  while (opt) { 

    printf("Enter with volume threshold\n");
    scanf("%d",&thres);
    getchar();

    t1     = iftTic();

    img[1] = iftVolumeClose(img[0],thres,ctree);
    
    t2     = iftToc();
    fprintf(stdout,"Volume closing executed in %f ms\n",iftCompTime(t1,t2));
    if (iftIs3DImage(img[1]))
      iftWriteImageByExt(img[1],"result.scn");
    else{
      if (iftIsColorImage(img[1]))
	iftWriteImageByExt(img[1],"result.ppm");
      else
	iftWriteImageByExt(img[1],"result.pgm");
    }
    iftDestroyImage(&img[1]);

    printf("Type 1 to continue and 0 to exit\n");
    scanf("%d",&opt);
    getchar();

  }

  iftDestroyImage(&img[0]);  
  iftDestroyCompTree(&ctree);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
