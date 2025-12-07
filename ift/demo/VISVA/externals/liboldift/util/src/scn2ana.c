
#include "ift.h"
#include <ctype.h>



int main(int argc, char **argv) 
{
  Scene *scn;


  if (argc!=3) {
    printf("Usage: %s [input.scn] [output.hdr or output.img]\n",argv[0]);
    exit(0);
  }

  scn = ReadScene(argv[1]);
  if (scn==NULL) {
    printf("Error: cannot read input file\n");
    exit(1);
  }
  WriteScene(scn,argv[2]);


  int len;
  char filename[300];
  strcpy(filename,argv[2]);
  len = strlen(filename);
  char ext[4];
  ext[0]=tolower(filename[len-3]);
  ext[1]=tolower(filename[len-2]);
  ext[2]=tolower(filename[len-1]);
  ext[3]='\0';
  if (strcmp(ext,"hdr")!=0 && strcmp(ext,"img")!=0) {
    exit(1);
  }
  char hdr_fname[300], img_fname[300];
  if (strcmp(ext,"hdr")==0) {
    strcpy(hdr_fname,filename);
    strcpy(img_fname,filename);
    img_fname[len-3]='i';
    img_fname[len-2]='m';
    img_fname[len-1]='g';
  }
  if (strcmp(ext,"img")==0) {
    strcpy(img_fname,filename);
    strcpy(hdr_fname,filename);
    hdr_fname[len-3]='h';
    hdr_fname[len-2]='d';
    hdr_fname[len-1]='r';
  }

  printf("%s written.\n",hdr_fname);
  printf("%s written.\n",img_fname);

  return 0;
}









