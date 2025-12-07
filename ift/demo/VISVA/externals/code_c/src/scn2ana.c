

#include "scn2ana.h"

struct _scn {
  int   W,H,D;
  float dx,dy,dz;
  int   bpp;
  int   maxval;
} scn;

struct _hdr1 {
  int        hdrlen;
  char       data_type[10];
  char       db_name[18];
  int        extents;
  short int  error;
  char       regular;
  char       hkey0;
} hdr1;

struct _hdr2 {
  short int  dim[8];
  short int  unused[7];
  short int  data_type;
  short int  bpp;
  short int  dim_un0;
  float      pixdim[8];
  float      zeroes[8];
  int        maxval;
  int        minval;
} hdr2;


void fgoto_1(int x,int y,int z,FILE *f, int bpp);
void write_hdr(char * hdr_name);
void read_scn(char * scn_name, char * hdr_name);


void fgoto_1(int x,int y,int z,FILE *f, int bpp) {
  long off;
  off = bpp * (x + (scn.W*(scn.H-1-y)) + (scn.W * scn.H) * z);
  fseek(f,off,SEEK_SET);  
}

void write_hdr(char * hdr_name) {
  int i;
  FILE *f;

  f = fopen(hdr_name,"w");
  if (!f) {
    fprintf(stderr,"** unable to open %s for writing.\n", hdr_name);
    exit(1);
  }

  memset(&hdr1, 0, sizeof(hdr1));
  memset(&hdr2, 0, sizeof(hdr2));

  for(i=0;i<8;i++) {
    hdr2.pixdim[i] = 0.0;
    hdr2.zeroes[i] = 0.0;
  }

  /* -- first header segment -- */

  hdr1.hdrlen  = 348;
  hdr1.regular = 'r';

  fio_abs_write_32(f, 0, 0, hdr1.hdrlen);
  fio_abs_write_zeroes(f, 4, 34);
  fio_abs_write_8(f, 38, hdr1.regular);
  fio_abs_write_8(f, 39, hdr1.hkey0);

  /* -- second header segment -- */

  hdr2.dim[0] = 4;
  hdr2.dim[1] = scn.W;
  hdr2.dim[2] = scn.H;
  hdr2.dim[3] = scn.D;
  hdr2.dim[4] = 1;

  hdr2.data_type = (scn.bpp == 16) ? 4 : 2;
  hdr2.bpp       = scn.bpp;
  hdr2.pixdim[0] = 1.0;
  hdr2.pixdim[1] = scn.dx;
  hdr2.pixdim[2] = scn.dy;
  hdr2.pixdim[3] = scn.dz;
  hdr2.maxval    = scn.maxval;
  hdr2.minval    = 0;

  for(i=0;i<8;i++)
    fio_abs_write_16(f, 40 + 0  + 2*i, 0, hdr2.dim[i]);
  for(i=0;i<7;i++)
    fio_abs_write_16(f, 40 + 16 + 2*i, 0, hdr2.unused[i]);

  fio_abs_write_16(f, 40 + 30, 0, hdr2.data_type);
  fio_abs_write_16(f, 40 + 32, 0, hdr2.bpp);
  fio_abs_write_16(f, 40 + 34, 0, hdr2.dim_un0);

  for(i=0;i<8;i++)
    fio_abs_write_float32(f, 40 + 36 + 4*i, 0, hdr2.pixdim[i]);
  for(i=0;i<8;i++)
    fio_abs_write_float32(f, 40 + 68 + 4*i, 0, hdr2.zeroes[i]);
 
  fio_abs_write_32(f, 40 + 100, 0, hdr2.maxval);
  fio_abs_write_32(f, 40 + 104, 0, hdr2.minval);

  /* -- third header segment (patient info) --- */

  fio_abs_write_zeroes(f, 148, 200);
  fclose(f);
}

void read_scn(char * scn_name, char * hdr_name) {
  FILE *f,*g;
  char *img_name;
  char line[256];
  char *buf,*buf2;
  int i,j,k,l,Bpp,cmax=0;
  int host_be = 0;
  short int *ibuf;
  int number,pos;

  k = 0x11110000;
  buf = (char *) (&k);
  if (buf[0] == 0x11) host_be = 1;

  f = fopen(scn_name,"r");
  if (!f) {
    fprintf(stderr,"** unable to open %s for reading.\n",scn_name);
    exit(1);
  }

  if (!fgets(line, 255, f))
    goto read_error;

  if (line[0]!='S' || line[1] !='C' || line[2] != 'N')
    goto format_error;

  if (fscanf(f,"%d %d %d\n",&(scn.W),&(scn.H),&(scn.D))!=3)
    goto format_error;

  if (fscanf(f,"%f %f %f\n",&(scn.dx),&(scn.dy),&(scn.dz))!=3)
    goto format_error;

  if (fscanf(f,"%d",&(scn.bpp))!=1)
    goto format_error;

  if (scn.bpp != 8 && scn.bpp != 16)
    goto format_error;

  pos = ftell(f);
  fseek(f,(pos+1)*sizeof(char),SEEK_SET); // +1 for the EOL \n character not included in last fscanf() call
  printf("%d %d %d %.2f %.2f %.2f %d\n",scn.W,scn.H,scn.D,scn.dx,scn.dy,scn.dz,scn.bpp);

  j = scn.H * scn.D;
  Bpp = scn.bpp / 8;

  buf = (char *) malloc( scn.W * Bpp );
  if (!buf)
    goto malloc_error;

  buf2 = (char *) malloc( scn.W * Bpp );
  if (!buf2)
    goto malloc_error;

  img_name = (char *) malloc(strlen(hdr_name)+1);
  strcpy(img_name, hdr_name);
  strcpy(img_name + strlen(hdr_name) - 3, "img");

  g = fopen(img_name,"w");
  if (!g) {
    fprintf(stderr,"** unable to open %s for writing.\n",
	    img_name);
    exit(1);
  }

  ibuf = (short int *) buf;

  /* copy pixel data */
  for(i=0;i<j;i++) {
    number = fread(buf, 1, scn.W * Bpp, f);
    if(number != (scn.W * Bpp))
      goto read_error;
    fgoto_1(0,i%(scn.H),i/(scn.H),g,Bpp);
    if (fwrite(buf, 1, scn.W * Bpp, g) != (scn.W * Bpp))
      goto write_error;

    if (scn.maxval < 0) {
      if (Bpp == 1) {
	for (k=0;k<scn.W;k++) {
	  l = (int) ((unsigned char) (buf[k]));
	  if (l > cmax) cmax = l;
	}
      } else {
	for (k=0;k<scn.W;k++) {
	  l = (int) (ibuf[k]);
	  if (host_be) l = fio_swab_16(l);
	  if (l > cmax) cmax = l;
	}
      }
    }
  }
  free(img_name);
  fclose(g);
  fclose(f);

  if (scn.maxval < 0)
    scn.maxval = cmax;

  return;

 format_error:
  fprintf(stderr,"** input file is not in SCN format.\n");
  exit(2);
 read_error:
  printf("number: %d, i: %d\n",number,i);
  fprintf(stderr,"** read error - %s.\n",strerror(errno));
  exit(2);
 write_error:
  fprintf(stderr,"** write error.\n");
  exit(2);
 malloc_error:
  fprintf(stderr,"** malloc error.\n");
  exit(3);

}


void write_analyze(char *scn_name, char *hdr_name, int maxval){
  if(maxval>=0)
    scn.maxval = maxval;
  else
    scn.maxval = -1;

  read_scn(scn_name, hdr_name);
  write_hdr(hdr_name);
}


