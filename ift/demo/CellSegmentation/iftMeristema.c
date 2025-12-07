#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage       *img[3];
  iftAdjRel      *A;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=4)
    iftError("Usage: iftMeristema <input.scn> <output.scn> <gt.scn>","main");

  img[0]   = iftReadImageByExt(argv[1]);    
  
  t1     = iftTic();

  img[1] = iftHBasins(img[0],30);
  img[2] = iftFastAreaOpen(img[1],10); 
  iftDestroyImage(&img[0]); 
  img[0] = iftThreshold(img[2],2,IFT_INFINITY_INT,1);
  iftDestroyImage(&img[1]); 
  A      = iftSpheric(1.5);
  img[1] = iftLabelComp(img[0],A);
  iftDestroyImage(&img[2]); 
  img[2] = iftReadImageByExt(argv[3]);

  t2     = iftToc();
  fprintf(stdout,"Cells detected in %f ms\n",iftCompTime(t1,t2));

  /* Monta uma tabela com as distancias entre os componentes (TDE) e
     outra com os rotulos do gt associados a cada componente. Verifica
     qual é a máxima distancia entre componentes associados a um mesmo
     rotulo do gt para fins de união deles. Mede o erro do
     reconhecimento, quando multiplos componentes são associados a um
     mesmo rótulo do gt ou um mesmo componente é associado a mais de
     um rótulo do gt */ 

  int  ncols=iftMaximumValue(img[2])+1, nrows=iftMaximumValue(img[1])+1;
  char comptb[nrows][ncols];

  for (int p=0; p < img[2]->n; p++) 
    if ((img[2]->val[p]!=0)&&(img[1]->val[p]!=0))
      comptb[img[1]->val[p]][img[2]->val[p]]=1;

  int nerror=0;
  
  for (int r=1; r < nrows; r++) {
    int n=0;
    for (int c=1; c < ncols; c++) {
      if (comptb[r][c])
	n++;
    }
    if (n > 1)
      nerror++;
  }

  printf("nrows %d ncols %d nerror %d \n",nrows,ncols,nerror);

  iftWriteImageByExt(img[1],argv[2]);

  iftDestroyImage(&img[0]);  
  iftDestroyImage(&img[1]);  
  iftDestroyImage(&img[2]);  
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

