#include "common.h"
#include "scene.h"

/* Computes the similarity between two labeled scenes */

int main(int argc, char **argv)
{
  Scene *scn1,*scn2;
  int   i,n,num_voxels_scn1=0,num_voxels_scn2=0,num_errors_scn1=0,num_errors_scn2=0;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc != 3) {
    fprintf(stderr,"usage: similarity <scene 1> <scene 2>\n");
    exit(-1);
  }
  scn1  = ReadScene(argv[1]);
  scn2  = ReadScene(argv[2]);

  if ((scn1->xsize==scn2->xsize)&&
      (scn1->ysize==scn2->ysize)&&
      (scn1->zsize==scn2->zsize)){
    n = scn1->xsize*scn1->ysize*scn1->zsize;
    for (i=0; i < n; i++) {
      if (scn1->data[i]!=0){
	num_voxels_scn1++;
	if (scn2->data[i]==0)
	  num_errors_scn1++;
      }
      if (scn2->data[i]!=0){
	num_voxels_scn2++;
	if (scn1->data[i]==0)
	  num_errors_scn2++;
      }
    }
    fprintf(stdout,"Similarity %f\n",1.0-((((float)num_errors_scn1/(float)num_voxels_scn1)+((float)num_errors_scn2/(float)num_voxels_scn2))/2.0));
  }else{
    fprintf(stderr,"scenes must have the same domain\n");
    exit(-1);    
  }
  
  DestroyScene(&scn1);
  DestroyScene(&scn2);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
