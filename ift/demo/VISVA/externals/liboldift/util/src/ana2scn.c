
#include "ift.h"




int main(int argc, char **argv) 
{
  Scene *scn;

  if (argc!=3) {
    printf("Usage: %s [input.hdr or input.img] [output.scn]\n",argv[0]);
    exit(0);
  }

  scn = ReadScene(argv[1]);
  if (scn==NULL) {
    printf("Error: cannot convert\n");
    exit(1);
  }
  WriteScene(scn,argv[2]);
  printf("%s written.\n",argv[2]);

  return 0;
}









