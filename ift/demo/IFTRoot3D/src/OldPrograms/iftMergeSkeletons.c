#include "ift.h"

int main(int argc, char *argv[]) 
{
  timer          *t1=NULL,*t2=NULL;
  iftImage       *label;
  iftImage       *skel[3];
  char            filename[200];
  int i, day, xsize, ysize, zsize, p, q, n;
  iftVoxel center[3], u, v; 

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=2)
    iftError("Usage must be: iftMergeSkeletons <basename>","main");

  // Reading csv file

  
  t1     = iftTic();
  
  for (i=0, day=3; i < 3; i++, day=day+3){
    sprintf(filename,"%s%d.csv",argv[1],day);
    skel[i] = iftCSVtoImage(filename);
  }

  xsize = MAX(MAX(skel[0]->xsize,skel[1]->xsize),skel[2]->xsize);
  ysize = MAX(MAX(skel[0]->ysize,skel[1]->ysize),skel[2]->ysize);
  zsize = MAX(MAX(skel[0]->zsize,skel[1]->zsize),skel[2]->zsize);

  for (i=0; i < 3; i++){
    center[i].x = center[i].y = center[i].z = n = 0;
    for (p=0; p < skel[i]->n; p++){
      if (skel[i]->val[p]!=0){
	u = iftGetVoxelCoord(skel[i],p);	
	n++;
	center[i].x += u.x;
	center[i].y += u.y;
	center[i].z += u.z;
      }
    }
    center[i].x /= n;
    center[i].y /= n;
    center[i].z /= n;
  }
  
  label = iftCreateImage(xsize,ysize,zsize);


  for (i=0; i < 3; i++){
    for (p=0; p < skel[i]->n; p++){
      if (skel[i]->val[p]!=0){
	u = iftGetVoxelCoord(skel[i],p);
	v.x = u.x - center[i].x + xsize/2;
	v.y = u.y - center[i].y + ysize/2;
	v.z = u.z - center[i].z + zsize/2;
	q   = iftGetVoxelIndex(label,v);
	switch(i) {
	case 0:	  
	  if (label->val[q]==0)
	    label->val[q] = 1;
	  break;
	case 1:	  
	  if (label->val[q]==0)
	    label->val[q] = 2;
	  break;
	case 2:	  
	  if (label->val[q]==0)
	    label->val[q] = 3;
	  break;
	}
      }
    }
  }

  t2     = iftToc();
  fprintf(stdout,"iftMergeSkeletons in %f ms\n",iftCompTime(t1,t2));

  for (i=0; i < 3; i++){
    iftDestroyImage(&skel[i]);
  }

  iftWriteImage(label,"label.scn");
  iftDestroyImage(&label);


  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



