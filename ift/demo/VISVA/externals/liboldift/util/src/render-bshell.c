#include "ift.h"


int MinimumBodyValue (Shell *shell)    
{

  Voxel v;
  int i,i1,i2,Imin=INT_MAX;
  
  if (shell->body)
    for (v.z = 0; v.z != shell->zsize; v.z ++) 
      for (v.y = 0; v.y != shell->ysize; v.y ++) {
        i = shell->Myz->tbrow[v.z] + v.y;     
        if (shell->Myz->val[i] >= 0) {	
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i];
	  for (i = i1; i != i2; i++) {	  
	    if (shell->body[i].val < Imin)
	      Imin = shell->body[i].val;
	  }        
	}
      }
  return(Imin);
}

int MaximumBodyValue (Shell *shell)    
{

  Voxel v;
  int i,i1,i2,Imax=0;
  
  if (shell->body)
    for (v.z = 0; v.z != shell->zsize; v.z ++) 
      for (v.y = 0; v.y != shell->ysize; v.y ++) {
        i = shell->Myz->tbrow[v.z] + v.y;     
        if (shell->Myz->val[i] >= 0) {	
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i];
	  for (i = i1; i != i2; i++) {	  
	    if (shell->body[i].val > Imax)
	      Imax = shell->body[i].val;
	  }        
	}
      }
  return(Imax);
}


int main(int argc, char **argv)
{
  timer tic,toc;
  FILE *ftime=fopen("time.txt","w");
  Shell *sh=NULL;
  Context *cxt=NULL;
  CImage *cimg=NULL,*simg=NULL;
  char filename[100];
  float tx,ty;
  int i,Imin,Imax;
 
  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc != 3) {
    fprintf(stderr,"usage: render-bshell <inputfile> <basename>\n");
    exit(-1);
  }

  /* Read bshell */

  sh     = ReadBodyShell(argv[1]);
  Imin   = MinimumBodyValue(sh);
  Imax   = MaximumBodyValue(sh);
  SetBodyShellVisibility (sh,Imin, Imax); 
  cxt = NewContext(sh->xsize,sh->ysize,sh->zsize);
  SetObjectColor(cxt,255,1.0,1.0,0.8);

  i = 100;
  gettimeofday(&tic,NULL);  
  for (tx=-85; tx<=85; tx=tx+30)
    for (ty=-180; ty < 180; ty=ty+30){
      SetAngles(cxt,tx,ty);       
      cimg   = CSWShellRendering(sh,cxt);
      simg   = CLinearStretch(cimg,10,200,0,255);
      sprintf(filename,"%s.%3d",argv[2],i);
      WriteCImage(simg,filename);
      DestroyCImage(&cimg);
      DestroyCImage(&simg);
      i++;
    }   
  gettimeofday(&toc,NULL);
  fprintf(ftime,"Rendering in %f milliseconds\n",((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/(i-100));

  DestroyShell(&sh);
  DestroyContext(&cxt);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  fclose(ftime);
  return(0);
}
