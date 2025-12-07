#include "ift.h"

iftImage *iftPanicleMask(iftImage *orig);
iftImage *iftSeedSegmentation(iftImage *orig);


iftImage *iftPanicleMask(iftImage *orig)
{
  iftAdjRel *A; 
  iftImage  *aux[2], *mask;

  /* Compute panicle mask */

  A      = iftCircular(1.0);
  aux[0] = iftErode(orig,A);
  aux[1] = iftThreshold(aux[0],0,ROUND(1.2*iftOtsu(orig)),255);
  iftDestroyImage(&aux[0]);
  iftDestroyAdjRel(&A);
  A      = iftCircular(sqrtf(2.0));
  mask   = iftSelectLargestComp(aux[1],A);
  iftDestroyImage(&aux[1]);
  iftDestroyAdjRel(&A);
  
  return(mask);
}

iftImage *iftSeedSegmentation(iftImage *orig)
{
  iftImage      *open,*basins,*label,*mask;
  iftAdjRel     *A;
  iftLabeledSet *markers=NULL;
  


  /* Compute panicle mask */
  
  mask = iftPanicleMask(orig);

  A=iftCircular(sqrtf(2.0));

  iftImage *edt=iftEuclDistTrans(mask,A,INTERIOR, NULL, NULL, NULL); 
  iftImage *hdome=iftHDomes(edt,121);
  iftWriteImageP2(hdome,"teste.pgm");
  iftDestroyImage(&edt);
  exit(0);

  /* Detect internal seed markers */

  open = iftOpenBin(mask,6.0); // 6.0
  iftDestroyImage(&mask); 

  label =  iftFastLabelComp(open,A); 
  iftDestroyImage(&open); 

  /* Compute basins */

  basins = iftImageBasins(orig,A);

  /* Segment seeds by watershed transform */


  markers=iftImageBorderLabeledSet(basins);
  for (int p=0; p < label->n; p++)
    if (label->val[p]>0) iftInsertLabeledSet(&markers,p,label->val[p]);

  iftDestroyImage(&label);
  label = iftWatershed(basins,A,markers,NULL);

  iftDestroyImage(&basins);
  iftDestroyLabeledSet(&markers);

  return(label);

}


int main(int argc, char *argv[]) 
{
  iftImage  *orig=NULL, *label=NULL;
  char       filename[400],command[500],*basename;
  char       ext[10],*pos;
  timer     *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=2)
    iftError("Usage: iftSeedSegm <panicle.jpg>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  basename = strtok(argv[1],".");

  if (strcmp(ext,"jpg")==0){
    sprintf(filename,"./images/%s.jpg",basename);
    sprintf(command,"convert %s temp.pgm",filename);
    system(command);
    orig  = iftReadImageP5("temp.pgm");
    sprintf(command,"rm -f temp.pgm");
    system(command);
  }else{
    printf("Invalid image format: %s\n",ext);
    exit(-1);
  }


  t1    = iftTic(); 
  
  label  = iftSeedSegmentation(orig); 

  t2    = iftToc();

  iftImage *temp = iftColorizeCompOverImage(orig,label);
  iftWriteImageP6(temp,"temp.ppm");
  sprintf(filename,"./images/%s_seeds.jpg",basename);
  sprintf(command,"convert temp.ppm %s",filename);
  system(command);
  sprintf(command,"rm -f temp.ppm");
  system(command);
  iftDestroyImage(&temp);

  iftDestroyImage(&orig);
  iftDestroyImage(&label);

  fprintf(stdout,"Seeds segmented in %f ms\n",iftCompTime(t1,t2));

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}




