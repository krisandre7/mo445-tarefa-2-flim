#include "common.h"
#include "radiometric3.h"

int main(int argc, char **argv)
{
  char filename[200];
  Scene *scn,*str;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc != 4) {
    fprintf(stderr,"usage: stretch <basename> <low> <high>\n");
    exit(-1);
  }
  sprintf(filename,"%s.scn",argv[1]);
  scn    = ReadScene(filename);
  str    = LinearStretch3(scn,MinimumValue3(scn),MaximumValue3(scn),atoi(argv[2]),atoi(argv[3]));
  sprintf(filename,"%s_str.scn",argv[1]);
  WriteScene(str,filename);
  DestroyScene(&scn);
  DestroyScene(&str);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
