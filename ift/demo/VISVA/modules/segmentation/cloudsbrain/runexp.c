
#include "oldift.h"
#include "compressed.h"

#include "shared.h"
#include "preproc.h"
#include "filelist.h"
#include "realarray.h"
#include "evaluation3.h"
#include "fuzzycloud3.h"
#include "seedmap3.h"
#include "scene_addons.h"


#define NOBJS 5

static char object[NOBJS][100] = {"brain",
				  "telencephalon",
				  "cerebellum",
				  "righthemisphere",
				  "lefthemisphere"};


Scene *Label2ObjectMask(Scene *label, char *name){
  Scene *mask=NULL;
  if(strcmp(name,"cerebellum")==0)
    mask = Threshold3(label, 1, 1);
  else if(strcmp(name,"telencephalon")==0)
    mask = Threshold3(label, 2, 3);
  else if(strcmp(name,"brain")==0)
    mask = Threshold3(label, 1, 3);
  else if(strcmp(name,"righthemisphere")==0)
    mask = Threshold3(label, 2, 2);
  else if(strcmp(name,"lefthemisphere")==0)
    mask = Threshold3(label, 3, 3);
  return mask;
}


int main(int argc, char **argv){
  int nimages,s,i,k,i_true;
  FileList *L[2];
  Scene *gtru,*segm,*scn,*diff,*tmp;
  Scene *gtru_bin,*segm_bin;
  char path[500];
  char filename[500];
  char command[500];
  real ja[NOBJS][200],di[NOBJS][200];
  real mean,stdev;
  FILE *fp_csv=NULL;
  char *bia_dir=NULL;

  bia_dir = getenv("BIA_DIR");
  if(bia_dir==NULL)
    Error((char*)"Environment variable BIA_DIR must be defined",
	  (char*)"runexp");


  //------- check number of parameters ----------
  if(argc != 2){
    fprintf(stdout,"usage: runexp <path>\n");
    fprintf(stdout,"path: Path where the file lists are.\n");
    fprintf(stdout,"Relation of expected files:\n");
    fprintf(stdout,"\tvolume.txt: image list of the original data.\n");
    fprintf(stdout,"\ttriobj.txt: label list of the objects.\n");
    exit(0);
  }

  //------- read and check parameters -----------
  strcpy(path, argv[1]);
  s = strlen(path);
  if(s>0 && path[s-1]=='/') path[s-1] = '\0';

  system("rm ./out/* -f");

  //------- read file lists ---------------------
  sprintf(filename,"%s/volume.txt",path);
  L[0] = ReadFileList(filename);
  sprintf(filename,"%s/triobj.txt",path);
  L[1] = ReadFileList(filename);
  nimages = L[0]->n;

  //------- glibc problem ---------------------
  //system("export MALLOC_CHECK_=0");
  //This will ignore the multi free() to particular memory.

  //------- create report file -------------------
  fp_csv = fopen("./out/report.csv","w");
  if(fp_csv == NULL) 
    Error("Cannot write CSV file","runexp");

  fprintf(fp_csv,"image#");
  for(k=0; k<NOBJS; k++){
    fprintf(fp_csv,"; %s",object[k]);
  }
  fprintf(fp_csv,"\n");

  //------- run exp ----------------------
  i_true = 0;
  for(i=0; i<nimages; i++){

    //------- convert images to LPS -----------
    gtru = ReadVolume(GetFile(L[1], i));
    scn  = ReadVolume(GetFile(L[0], i));

    tmp  = ChangeOrientationToLPS(gtru, (char *)"PIR");
    DestroyScene(&gtru);
    gtru  = tmp;

    tmp  = ChangeOrientationToLPS(scn, (char *)"PIR");
    DestroyScene(&scn);
    scn  = tmp;
    sprintf(filename,"image.scn");    
    WriteVolume(scn, filename);

    //------- segment volume i ----------------
    sprintf(command,"%s/bin/cloudsbrain_bin %s",bia_dir,filename);
    system(command);

    //------- evaluate results ----------------
    sprintf(filename,"segm_brain.scn");
    segm = ReadVolume(filename);
 
    sprintf(filename,"./out/gtruth%03d_triobj.scn.bz2",i+1);
    WriteVolume(gtru, filename);
    
    sprintf(filename,"./out/segm%03d_triobj.scn.bz2",i+1);
    WriteVolume(segm, filename);

    diff = XOr3(segm, gtru);
    sprintf(filename,"./out/diff%03d_triobj.scn.bz2",i+1);
    WriteVolume(diff, filename);

    fprintf(fp_csv,"%02d",i+1);

    for(k=0; k<NOBJS; k++){  
      segm_bin = Label2ObjectMask(segm, object[k]);
      gtru_bin = Label2ObjectMask(gtru, object[k]);
      ja[k][i_true] = JaccardSimilarity3(segm_bin, gtru_bin);
      di[k][i_true] = DiceSimilarity3(segm_bin, gtru_bin);
      printf("%s%02d",object[k],i+1);
      PrintDots(16-strlen(object[k]));
      printf("(jaccard=%f, dice=%f)\n",ja[k][i_true],di[k][i_true]);
      DestroyScene(&segm_bin);
      DestroyScene(&gtru_bin);

      fprintf(fp_csv,"; %f",di[k][i_true]);
    }
    fprintf(fp_csv,"\n");

    DestroyScene(&diff);
    DestroyScene(&scn);
    DestroyScene(&segm);
    DestroyScene(&gtru);
    i_true++;
  }

  //------- statistics --------------------------------
  for(k=0; k<NOBJS; k++){
    printf("Jaccard %s",object[k]);
    PrintDots(16-strlen(object[k]));
    RealArrayStatistics(ja[k], i_true, &mean, &stdev);
    printf("mean,stdev = (%f,%f)\n",mean,stdev);

    printf("Dice    %s",object[k]);
    PrintDots(16-strlen(object[k]));
    RealArrayStatistics(di[k], i_true, &mean, &stdev);
    printf("mean,stdev = (%f,%f)\n",mean,stdev);
  }

  DestroyFileList(&L[0]);
  DestroyFileList(&L[1]);
  fclose(fp_csv);

  return(0);
}

