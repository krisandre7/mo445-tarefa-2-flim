#include "ift.h"

iftAdjRel *GetAdjacencyRelation(iftImage *img, float radius)
{
  iftAdjRel *A;
  
  if (iftIs3DImage(img))
    A = iftSpheric(radius);
  else
    A = iftCircular(radius);

  return(A);
}

void PrintErrorMessage()
{
  printf("Types of morphological operation:\n");
  printf("DILATION     =0\n");
  printf("EROSION      =1\n");
  printf("CLOSING      =2\n");
  printf("OPENING      =3\n");
  printf("HCLOSING     =4\n");
  printf("HOPENING     =5\n");
  printf("AREACLOSING  =6\n");
  printf("AREAOPENING  =7\n");
  printf("VOLUMECLOSING=8\n");
  printf("VOLUMEOPENING=9\n");
  printf("CLOSEBASINS  =10\n");
  printf("OPENDOMES    =11\n");
  printf("MORPHGRAD    =12\n");
  printf("CLOSEREC     =13\n");
  printf("OPENREC      =14\n");
  iftError("Usage: iftMorph <image.* [ppm,pgm,scn,png]> <type of morphological operation> <parameter of the operator [adj. relation, area threshold, depth]> <output.[pgm,ppm,scn,png]>","main");
}

typedef enum ift_Morph_Operations {
  DILATION     =0,
  EROSION      =1,
  CLOSING      =2,
  OPENING      =3,
  HCLOSING     =4,
  HOPENING     =5,
  AREACLOSING  =6,
  AREAOPENING  =7,
  VOLUMECLOSING=8,
  VOLUMEOPENING=9,
  CLOSEBASINS  =10,
  OPENDOMES    =11,
  MORPHGRAD    =12,
  CLOSEREC     =13,
  OPENREC      =14,  
} iftMorphOperations;

int main(int argc, char *argv[]) 
{
  iftImage  *img[3];
  iftAdjRel *A = NULL;
  timer     *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=5){
    PrintErrorMessage();
  }

  
  img[0] = iftReadImageByExt(argv[1]);
  img[1]=img[2]=NULL;
  
  t1     = iftTic();

  switch(atoi(argv[2])){

  case DILATION:
    if (iftIsBinaryImage(img[0])){
      iftSet   *S=NULL;
      iftImage *aux1 = iftAddFrame(img[0],iftRound(atof(argv[3]))+5,0);
      iftImage *aux2 = iftDilateBin(aux1,&S,atof(argv[3]));    
      img[2]         = iftRemFrame(aux2,iftRound(atof(argv[3]))+5);
      iftDestroyImage(&aux1);
      iftDestroyImage(&aux2);
      iftDestroySet(&S);
    } else{
      A = GetAdjacencyRelation(img[0],atof(argv[3]));
      img[2] = iftDilate(img[0],A,NULL);
      iftDestroyAdjRel(&A);
    }
    break;
    
  case EROSION:
    if (iftIsBinaryImage(img[0])){
      iftSet   *S=NULL;
      iftImage *aux1 = iftAddFrame(img[0],iftRound(atof(argv[3]))+5,0);
      iftImage *aux2 = iftErodeBin(aux1,&S,atof(argv[3]));    
      img[2]         = iftRemFrame(aux2,iftRound(atof(argv[3]))+5);
      iftDestroyImage(&aux1);
      iftDestroyImage(&aux2);
      iftDestroySet(&S);
    } else{
      A = GetAdjacencyRelation(img[0],atof(argv[3]));
      img[2] = iftErode(img[0],A,NULL);
      iftDestroyAdjRel(&A);
    }
    break;
    
  case CLOSING:
    if (iftIsBinaryImage(img[0]))
      img[2] = iftCloseBin(img[0],atof(argv[3]));    
    else{
      A = GetAdjacencyRelation(img[0],atof(argv[3]));
      img[2] = iftClose(img[0],A,NULL);    
      iftDestroyAdjRel(&A);
    }
    break;

  case OPENING:
    if (iftIsBinaryImage(img[0]))
      img[2] = iftOpenBin(img[0],atof(argv[3]));    
    else{
      A = GetAdjacencyRelation(img[0],atof(argv[3]));
      img[2] = iftOpen(img[0],A,NULL);    
      iftDestroyAdjRel(&A);
    }
    break;
    
  case HCLOSING:
    img[2] = iftHClose(img[0],atoi(argv[3]),NULL);    
    break;

  case HOPENING:
    img[2] = iftHOpen(img[0],atoi(argv[3]),NULL);    
    break;

  case AREACLOSING:
    img[2] = iftFastAreaClose(img[0],atoi(argv[3]));    
    break;

  case AREAOPENING:
    img[2] = iftFastAreaOpen(img[0],atoi(argv[3]));    
    break;

  case VOLUMECLOSING:
    img[2] = iftVolumeClose(img[0],atoi(argv[3]),NULL);    
    break;

  case VOLUMEOPENING:

    img[2] = iftVolumeOpen(img[0],atoi(argv[3]),NULL);    
    break;

  case CLOSEBASINS:

    img[2] = iftCloseBasins(img[0],NULL,NULL);    
    break;

  case OPENDOMES:

    img[2] = iftOpenDomes(img[0],NULL,NULL);    
    break;

  case MORPHGRAD:    
    A      = GetAdjacencyRelation(img[0],atof(argv[3]));
    img[2] = iftMorphGrad(img[0],A);    
    iftDestroyAdjRel(&A);
    break;

  case CLOSEREC:
    if (iftIsBinaryImage(img[0]))
      img[2] = iftCloseRecBin(img[0],atof(argv[3]));
    else{
      A = GetAdjacencyRelation(img[0],atof(argv[3]));
      img[2] = iftCloseRec(img[0],A,NULL);
      iftDestroyAdjRel(&A);
    }
    break;
    
  case OPENREC:
    if (iftIsBinaryImage(img[0]))
      img[2] = iftOpenRecBin(img[0],atof(argv[3]));
    else{
      A = GetAdjacencyRelation(img[0],atof(argv[3]));
      img[2] = iftOpenRec(img[0],A,NULL);
      iftDestroyAdjRel(&A);
    }
    break;
    
  default:
    PrintErrorMessage();
  }

  t2     = iftToc();
  fprintf(stdout,"Morph. operation in %f ms\n",iftCompTime(t1,t2));

  iftDestroyImage(&img[0]);
  if (img[1]!=NULL) 
    iftDestroyImage(&img[1]);
    
  iftWriteImageByExt(img[2],argv[4]);
  iftDestroyImage(&img[2]);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);

}
