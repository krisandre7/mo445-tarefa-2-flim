#include "ift.h"


int main(int argc, char **argv) {
  char filename[100];
  Scene *scn,*msk,*nscn;
  AdjRel3 *A;

  if (argc!=3) {
    fprintf(stderr,"usage: iftthres3 <basename> <adj. radius>\n");
    fprintf(stderr,"basename: file name without .scn extension\n");
    exit(-1);
  }

  sprintf(filename,"%s.scn",argv[1]);
  scn = ReadScene(filename);
  A     = Spheric(atof(argv[2]));

  msk   = iftThres3(scn,A);

  nscn  = Mult3(scn,msk);
  sprintf(filename,"%s-new.scn",argv[1]);
  WriteScene(nscn,filename);


  DestroyScene(&scn);
  DestroyScene(&msk);
  DestroyScene(&nscn);
  DestroyAdjRel3(&A);
  return(0);
}

