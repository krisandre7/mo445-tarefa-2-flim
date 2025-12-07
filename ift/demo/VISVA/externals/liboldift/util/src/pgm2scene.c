#include "common.h"
#include "scene.h"

int main(int argc, char **argv)
{
  FILE *fp=NULL; 
  char filename[200],type[10];
  unsigned char *data8;
  int i,j,n,v,first,last,xsize,ysize,zsize;
  Scene *scn;
  float R,G,B,Y;
  int *data32;

  /*--------------------------------------------------------*/
#ifndef _WIN32
  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif
  /*--------------------------------------------------------*/

  if (argc != 7) {
    fprintf(stderr,"usage: pgm2scene <basename> <first> <last> <dx> <dy> <dz>\n");
    exit(-1);
  }
  first = atoi(argv[2]);
  last  = atoi(argv[3]);
  sprintf(filename,"%s.%3d",argv[1],first);
  fp = fopen(filename,"r");
  if (fp == NULL) {
    Error(MSG2,"pgm2scene");
  }
  fscanf(fp,"%s",type); 
  fscanf(fp,"%d %d",&xsize,&ysize);
  n     = xsize*ysize;
  zsize = last - first + 1;
  scn   = CreateScene(xsize,ysize,zsize);
  scn->dx = atof(argv[4]);
  scn->dy = atof(argv[5]);
  scn->dz = atof(argv[6]);
  fclose(fp);

  if (strcmp(type,"P5")==0){    
    data8 = AllocUCharArray(n);
    data32 = scn->data;
    for (i=first; i <= last; i++) {
      sprintf(filename,"%s.%3d",argv[1],i);
      fp = fopen(filename,"r");
      if (fp == NULL) {
	Error(MSG2,"pgm2scene");
      }
      fscanf(fp,"%s",type); 
      fscanf(fp,"%d %d",&xsize,&ysize);
      fscanf(fp,"%d",&v);
      fgetc(fp);
      fread(data8,sizeof(uchar),n,fp);
      for (j=0; j < n; j++) 
	data32[j] = (int) data8[j];
      data32 = data32 + n;
      fclose(fp);
    }
    free(data8);
  } else {
    if (strcmp(type,"P2")==0){   
      data32 = scn->data;
      for (i=first; i <= last; i++) {
	sprintf(filename,"%s.%3d",argv[1],i);
	fp = fopen(filename,"r");
	if (fp == NULL) {
	  Error(MSG2,"pgm2scene");
	}
	fscanf(fp,"%s",type); 
	fscanf(fp,"%d %d",&xsize,&ysize);
	fscanf(fp,"%d",&v);
	for (j=0; j < n; j++){ 
	  fscanf(fp,"%d",&v);
	  data32[j] = (int) v;
	}
	data32 = data32 + n;
	fclose(fp);
      }
    } 
    else {
      if (strcmp(type,"P6")==0){   
	data32 = scn->data;
	for (i=first; i <= last; i++) {
	  sprintf(filename,"%s.%3d",argv[1],i);
	  fp = fopen(filename,"r");
	  if (fp == NULL) {
	    Error(MSG2,"pgm2scene");
	  }
	  fscanf(fp,"%s",type); 
	  fscanf(fp,"%d %d",&xsize,&ysize);
	  fscanf(fp,"%d",&v);
	  for (j=0; j < n; j++){ 
	    R = (float)fgetc(fp);
	    G = (float)fgetc(fp);
	    B = (float)fgetc(fp);
	    Y   = 0.257*R+0.504*G+0.098*B;
	    data32[j] = (int)Y;
	  }
	  data32 = data32 + n;
	  fclose(fp);
	}
      } 
      else {
	fprintf(stderr,"File format must be P2/P5/P6\n");
	exit(-1);
      }
    }
  }

  sprintf(filename,"%s.scn",argv[1]);
  WriteScene(scn,filename);
  DestroyScene(&scn);

  /* ---------------------------------------------------------- */
#ifndef _WIN32
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   
#endif

  return(0);
}
