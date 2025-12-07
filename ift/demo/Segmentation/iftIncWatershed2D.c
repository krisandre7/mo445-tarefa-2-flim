#include "ift.h"

int main(int argc, char *argv[]) 
{
  iftImage        *img=NULL,*basins=NULL;
  iftImageForest  *fst=NULL;
  iftImage        *marker=NULL;
  iftLabeledSet   *seed=NULL;
  iftAdjRel       *A=NULL;
  char             ext[10],*pos;
  char filename[100];
  iftImage *tmp_label=NULL;
  int iteration=1, finish, mode, prev_mode=NIL;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=4)
    iftError("Usage: iftIncWatershed2D <image.[pgm,ppm]> <spatial_radius> <volume_thres>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);
  
  if (strcmp(ext,"pgm")==0){
    img   = iftReadImageP5(argv[1]);    
  }else{
    if (strcmp(ext,"ppm")==0){
      img   = iftReadImageP6(argv[1]);    
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }
  
  t1 = iftTic();

  /* the operation is connected for the topology defined by A: A must
     be the same in all operators (including
     iftVolumeClose?). Otherwise, this is not a connected operation in
     the desired topology. */

  A      = iftCircular(atof(argv[2]));
  printf("computing basins\n");
  basins = iftImageBasins(img,A);
  printf("computing volume closing\n");
  marker = iftVolumeClose(basins,atof(argv[3]));
  fst    = iftCreateImageForest(basins,A);


  fprintf(stdout,"Type 0 to continue and 1 to exit\n");
  fscanf(stdin,"%d",&finish);

  while (!finish) {

    fprintf(stdout,"Chose operation mode: 0 - select object trees, 1 - Inc IFT\n");
    fscanf(stdin,"%d",&mode);

    fprintf(stdout,"Enter with a seed file\n");
    fscanf(stdin,"%s",filename);
    pos = strrchr(filename,'.') + 1;
    sscanf(pos,"%s",ext);
    if (strcmp(ext,"txt")==0){
      seed=iftReadSeeds2D(filename,basins);
    }else{
      printf("Invalid file format: %s\n",ext);
      exit(-1);
    }
      
    if (mode==0){
      printf("selecting object trees\n");
      switch (prev_mode){
      case NIL:
	printf("computing watergray\n");
	iftWaterGrayForest(fst,marker);
	//iftRelabelBorderTreesAsBkg(fst);
	tmp_label = iftSwitchTreeLabels(fst,seed,NULL);
	break;
      case 0:
	iftSwitchTreeLabels(fst,seed,tmp_label);
	break;
      case 1:
	printf("computing watergray\n");
	iftDestroyImageForest(&fst);
	fst    = iftCreateImageForest(basins,A);
	iftWaterGrayForest(fst,marker);
	//iftRelabelBorderTreesAsBkg(fst);
	tmp_label = iftSwitchTreeLabels(fst,seed,NULL);
      }
      sprintf(filename,"labels%d-%d.pgm",mode,iteration);
      iftCopyVoxelSize(img,tmp_label);
      iftWriteImageP2(tmp_label,filename);
      prev_mode = 0;
    }else{ //mode==1

      if (prev_mode==0){
	iftDestroyImage(&fst->label);
	fst->label = tmp_label;
	tmp_label  = NULL;
	prev_mode=1;
      }
    
      printf("computing  incremental IFT\n");
      iftIncWatershed(fst,seed);
      sprintf(filename,"labels%d-%d.pgm",mode,iteration);
      iftCopyVoxelSize(img,fst->label);
      iftWriteImageP2(fst->label,filename); 
      prev_mode = 1;
    }      

    iteration++;
    iftDestroyLabeledSet(&seed);

    fprintf(stdout,"Type 0 to continue and 1 to exit\n");
    fscanf(stdin,"%d",&finish);
  }

  t2     = iftToc(); 

  fprintf(stdout,"watergray+Inc IFT in %f ms\n",iftCompTime(t1,t2));

  iftDestroyImageForest(&fst);  
  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);  
  iftDestroyImage(&marker);  
  iftDestroyImage(&basins);  
  iftDestroyImage(&tmp_label);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

