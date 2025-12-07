
#include <ctype.h>
#include "scene.h"


typedef struct {
  int hdrlen;
  int bpp;
  int dimensions;
  int W,H,D,T;
  float dx,dy,dz;
  int be_hint;
  int dt;
} AnalyzeHdr;

typedef struct {
  int        hdrlen;
  char       data_type[10];
  char       db_name[18];
  int        extents;
  short int  error;
  char       regular;
  char       hkey0;
} Ana_Hdr1;

typedef struct {
  short int  dim[8];
  short int  unused[7];
  short int  data_type;
  short int  bpp;
  short int  dim_un0;
  float      pixdim[8];
  float      zeroes[8];
  int        maxval;
  int        minval;
} Ana_Hdr2;



int ana_fio_swab_16(int val);
int ana_fio_swab_32(int val);
int   ana_fio_abs_read_8(FILE *f, long offset);
int   ana_fio_abs_read_16(FILE *f, long offset, int is_big_endian);
int   ana_fio_abs_read_32(FILE *f, long offset, int is_big_endian);
float ana_fio_abs_read_float32(FILE *f, long offset, int is_big_endian);
int   ana_fio_read_8(FILE *f);
int   ana_fio_read_16(FILE *f, int is_big_endian);
int   ana_fio_read_32(FILE *f, int is_big_endian);
float ana_fio_read_float32(FILE *f, int is_big_endian);
int  ana_fio_abs_write_8(FILE *f, long offset, int val);
int  ana_fio_abs_write_16(FILE *f, long offset, int is_big_endian, int val);
int  ana_fio_abs_write_32(FILE *f, long offset, int is_big_endian, int val);
int  ana_fio_abs_write_float32(FILE *f, long offset, int is_big_endian, float val);
int  ana_fio_abs_write_zeroes(FILE *f, long offset, int n);
int  ana_fio_write_8(FILE *f, int val);
int  ana_fio_write_16(FILE *f, int is_big_endian, int val);
int  ana_fio_write_32(FILE *f, int is_big_endian, int val);
int  ana_fio_write_float32(FILE *f, int is_big_endian, float val);
int  ana_fio_write_zeroes(FILE *f, int n);


static int ana_fio_big_endian = -1;

void ana_fio_init() {
  short int x;
  char *p;

  x = 0x0001;
  p = (char *) (&x);

  if ((*p) == 0)
    ana_fio_big_endian = 1;
  else
    ana_fio_big_endian = 0;
}

int ana_fio_swab_16(int val) {
  int x;
  val &= 0xffff;
  x  = (val & 0xff) << 8;
  x |= (val >> 8);
  return x;
}

int ana_fio_swab_32(int val) {
  int x;
  val &= 0xffffffff;
  x  = (val & 0xff) << 24;
  x |= ((val >> 8) & 0xff) << 16;
  x |= ((val >> 16) & 0xff) << 8;
  x |= ((val >> 24) & 0xff);
  return x;
}

void *SwapEndian(void* Addr,   int Nb) {
	static char Swapped[16];
	switch (Nb) {
		case 2:	Swapped[0]=*((char*)Addr+1);
				Swapped[1]=*((char*)Addr  );
				break;
		case 3:	// As far as I know, 3 is used only with RGB images
				Swapped[0]=*((char*)Addr+2);
				Swapped[1]=*((char*)Addr+1);
				Swapped[2]=*((char*)Addr  );
				break;
		case 4:	Swapped[0]=*((char*)Addr+3);
				Swapped[1]=*((char*)Addr+2);
				Swapped[2]=*((char*)Addr+1);
				Swapped[3]=*((char*)Addr  );
				break;
		case 8:	Swapped[0]=*((char*)Addr+7);
				Swapped[1]=*((char*)Addr+6);
				Swapped[2]=*((char*)Addr+5);
				Swapped[3]=*((char*)Addr+4);
				Swapped[4]=*((char*)Addr+3);
				Swapped[5]=*((char*)Addr+2);
				Swapped[6]=*((char*)Addr+1);
				Swapped[7]=*((char*)Addr  );
				break;
		case 16:Swapped[0]=*((char*)Addr+15);
				Swapped[1]=*((char*)Addr+14);
				Swapped[2]=*((char*)Addr+13);
				Swapped[3]=*((char*)Addr+12);
				Swapped[4]=*((char*)Addr+11);
				Swapped[5]=*((char*)Addr+10);
				Swapped[6]=*((char*)Addr+9);
				Swapped[7]=*((char*)Addr+8);
				Swapped[8]=*((char*)Addr+7);
				Swapped[9]=*((char*)Addr+6);
				Swapped[10]=*((char*)Addr+5);
				Swapped[11]=*((char*)Addr+4);
				Swapped[12]=*((char*)Addr+3);
				Swapped[13]=*((char*)Addr+2);
				Swapped[14]=*((char*)Addr+1);
				Swapped[15]=*((char*)Addr  );
				break;
	}
	return (void*)Swapped;
}

#define SWAP_FLOAT(Var) *(float*)SwapEndian((void*)&(Var), sizeof(float))

int   ana_fio_abs_read_8(FILE *f, long offset) {
  if (fseek(f,offset,SEEK_SET)!=0) return -1;
  return(ana_fio_read_8(f));
}

int ana_fio_abs_read_16(FILE *f, long offset, int is_big_endian) {
  if (fseek(f,offset,SEEK_SET)!=0) return -1;
  return(ana_fio_read_16(f,is_big_endian));
}

int ana_fio_abs_read_32(FILE *f, long offset, int is_big_endian) {
  if (fseek(f,offset,SEEK_SET)!=0) return -1;
  return(ana_fio_read_32(f,is_big_endian));
}

float ana_fio_abs_read_float32(FILE *f, long offset, int is_big_endian) {
  union {
    int i;
    float f;
  } val;
  val.i = ana_fio_abs_read_32(f,offset,is_big_endian);
  return val.f;
}

int   ana_fio_read_8(FILE *f) {
  char val;

  if (ana_fio_big_endian < 0) ana_fio_init();
  if (fread(&val,1,1,f)!=1) return -1;

  return((int)val);
}

int   ana_fio_read_16(FILE *f, int is_big_endian) {
  short int val;

  if (ana_fio_big_endian < 0) ana_fio_init();
  if (fread(&val,1,2,f)!=2) return -1;

  if ( (is_big_endian!=0) != (ana_fio_big_endian!=0) )
    val = (short int) ana_fio_swab_16(val);
  return ((int)val);
}

int   ana_fio_read_32(FILE *f, int is_big_endian) {
  int val;

  if (ana_fio_big_endian < 0) ana_fio_init();
  if (fread(&val,1,4,f)!=4) return -1;

  if ( (is_big_endian!=0) != (ana_fio_big_endian!=0) )
    val = (short int) ana_fio_swab_32(val);
  return val;
}

float ana_fio_read_float32(FILE *f, int is_big_endian) {
  int val;
  void *vp;
  float *p, q;
  val = ana_fio_read_32(f,is_big_endian);
  vp = (void *) (&val);
  p = (float *) (vp);
  q = *p;
  return q;
}

int ana_fio_abs_write_8(FILE *f, long offset, int val) {
  if (fseek(f,offset,SEEK_SET)!=0) return -1;
  return(ana_fio_write_8(f,val));
}

int ana_fio_abs_write_16(FILE *f, long offset, int is_big_endian, int val) {
  if (fseek(f,offset,SEEK_SET)!=0) return -1;
  return(ana_fio_write_16(f,is_big_endian,val));
}

int ana_fio_abs_write_32(FILE *f, long offset, int is_big_endian, int val) {
  if (fseek(f,offset,SEEK_SET)!=0) return -1;
  return(ana_fio_write_32(f,is_big_endian,val));
}

int ana_fio_abs_write_float32(FILE *f, long offset, int is_big_endian, float val) {
  if (fseek(f,offset,SEEK_SET)!=0) return -1;
  return(ana_fio_write_float32(f,is_big_endian,val));
}

int   ana_fio_write_8(FILE *f, int val) {
  char v;
  if (ana_fio_big_endian < 0) ana_fio_init();
  v = (char) (val & 0xff);
  if (fwrite(&v,1,1,f)!=1) return -1;
  return 0;
}

int   ana_fio_write_16(FILE *f, int is_big_endian, int val) {
  short int v;
  if (ana_fio_big_endian < 0) ana_fio_init();
  v = (short int) (val & 0xffff);
  if ( (is_big_endian!=0) != (ana_fio_big_endian!=0) )
    v = (short int) ana_fio_swab_16(v);
  if (fwrite(&v,1,2,f)!=2) return -1;
  return 0;
}

int   ana_fio_write_32(FILE *f, int is_big_endian, int val) {
  int v = val;
  if (ana_fio_big_endian < 0) ana_fio_init();
  if ( (is_big_endian!=0) != (ana_fio_big_endian!=0) )
    v = ana_fio_swab_32(v);
  if (fwrite(&v,1,4,f)!=4) return -1;
  return 0;
}

int ana_fio_write_float32(FILE *f, int is_big_endian, float val) {
  float v, *w;
  int i, *j;
  void *vp;
  if (ana_fio_big_endian < 0) ana_fio_init();
  v = val;
  if ( (is_big_endian!=0) != (ana_fio_big_endian!=0) ) {
    vp = (void *) (&v);
    j = (int *) (vp);
    i = *j;
    i = ana_fio_swab_32(i);
    w = (float *) j;
    v = *w;
  }
  if (fwrite(&v,1,4,f)!=4) return -1;
  return 0;
}

int  ana_fio_abs_write_zeroes(FILE *f, long offset, int n) {
  if (fseek(f,offset,SEEK_SET)!=0) return -1;
  return(ana_fio_write_zeroes(f,n));
}

int  ana_fio_write_zeroes(FILE *f, int n) {
  while (n>0) {
    if (n>=4) {
      if (ana_fio_write_32(f, 0, 0) != 0) return -1;
      n -= 4;
    } else if (n>=2) {
      if (ana_fio_write_16(f, 0, 0) != 0) return -1;
      n -= 2;
    } else {
      if (ana_fio_write_8(f, 0) != 0) return -1;
      n--;
    }
  }
  return 0;
}











Scene* ReadScene_Analyze(char filename[])
{
  int len;
  char ext[4];
  char hdr_fname[300], img_fname[300];
  AnalyzeHdr hdr;
  FILE *f;
  int obpp = -1;
  int be = 0;
  Scene *scn;
  float *data;
  Voxel v;
  int p,n=0;
  unsigned char val8;
  short int val16;
  long int val32;
  float fval32;
  float fmin = FLT_MAX;
  float fmax = -FLT_MAX;
  float fdelta;
  
  len = strlen(filename);
  ext[0]=tolower(filename[len-3]);
  ext[1]=tolower(filename[len-2]);
  ext[2]=tolower(filename[len-1]);
  ext[3]='\0';
  if (strcmp(ext,"hdr")!=0 && strcmp(ext,"img")!=0) 
    return NULL;

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

  // READING HDR FILE  ------------------------
  f = fopen(hdr_fname,"r");
  if (!f) {
    fprintf(stderr,"** unable to open %s for reading.\n", hdr_fname);
    return NULL;
  }

  hdr.hdrlen = ana_fio_abs_read_32(f,0,be);
  if (hdr.hdrlen != 348) {
    be = 1;
    hdr.hdrlen = ana_fio_abs_read_32(f,0,be);
  }
  if (hdr.hdrlen!=348) {
    fprintf(stderr,"** This is not an Analyze header!\n");
    return NULL;
  }
  hdr.dt           = ana_fio_abs_read_16(f,40+30,be);
  hdr.bpp          = ana_fio_abs_read_16(f,40+32,be);
  hdr.dimensions   = ana_fio_abs_read_16(f,40,be);
  hdr.W            = ana_fio_abs_read_16(f,40+2,be);
  hdr.H            = ana_fio_abs_read_16(f,40+4,be);
  hdr.D            = ana_fio_abs_read_16(f,40+6,be);
  hdr.T            = ana_fio_abs_read_16(f,40+8,be);

  hdr.dx           = ana_fio_abs_read_float32(f,40+36+4,be);
  hdr.dy           = ana_fio_abs_read_float32(f,40+36+8,be);
  hdr.dz           = ana_fio_abs_read_float32(f,40+36+12,be);

  fclose(f);

  if (hdr.dimensions == 4 && hdr.T != 1) {
    fprintf(stderr,"** This file has a 3D time series, I can't convert it.\n");
    return NULL;
  }
  if (hdr.dimensions < 3 || hdr.dimensions > 4) {
    fprintf(stderr,"** This file is not a 3D volume, I can't convert it.\n");
    return NULL;
  }
/*   if (hdr.dt==16) { */
/*     fprintf(stderr,"** Warning: cannot read floating-point file.\n"); */
/*     return NULL; */
/*   } */
  if (obpp < 0) obpp = hdr.bpp;
  if (hdr.bpp != 8 && hdr.bpp != 16 && hdr.bpp != 32) {
    fprintf(stderr,"** This is %d-bit data, only 8-, 16-bit and 32-bit data are supported.\n",hdr.bpp);
    return NULL;
  }
  hdr.be_hint = be;

  // READING THE IMG FILE --------------------
  
  f = fopen(img_fname, "rb");
  if (!f) {
    fprintf(stderr,"** unable to open %s file.\n",img_fname);
    return NULL;
  }
  
  scn = CreateScene(hdr.W, hdr.H, hdr.D);
  data = ( float* ) calloc( hdr.W * hdr.H * hdr.D, sizeof( float ) );
    if (scn==NULL) {
      fprintf(stderr,"** CreateScene() error.\n");
      return NULL;
    }
  SetVoxelSize(scn, hdr.dx, hdr.dy, hdr.dz);
  
  for (v.z=0; v.z < scn->zsize; v.z++)
    for (v.y=scn->ysize-1; v.y >= 0; v.y--) // reverse order
      for (v.x=0; v.x < scn->xsize; v.x++) {
	p = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (hdr.bpp==8) {
	  n=fread(&val8,sizeof(unsigned char),1,f);
	  scn->data[p]=(int)val8;
	}
	if (hdr.bpp==16) {
	  n=fread(&val16,sizeof(short int),1,f);
	  if (hdr.be_hint) val16 = (short int) ana_fio_swab_16(val16);
	  scn->data[p]=(int)val16;
	}
	if (hdr.bpp==32) {
	  if (hdr.dt==8) { // integer 
	    n=fread(&val32,sizeof(long int),1,f);
	    if (hdr.be_hint) val32 = (long int) ana_fio_swab_32(val32);
	    scn->data[p]=(int)val32;
	  } 
	  else {
	    n=fread(&fval32,sizeof(float),1,f);
	    if (hdr.be_hint) 
	      fval32 = SWAP_FLOAT(fval32); // (float) ana_fio_swab_f32(fval32);
	    data[ p ]= fval32;
	    if( fmin > fval32 )
	      fmin = fval32;
	    if( fmax < fval32 )
	      fmax = fval32;
	  }
      	}
	if (n!=1) {
	  fprintf(stderr,"** error reading file %s.\n",img_fname);
	  DestroyScene(&scn);
	  return NULL;
	}
      }
  if (hdr.dt > 8) {    // != 8 ) {
    fdelta = fmax - fmin;
    //printf( "fdelta:%f, fmin:%f, fmax:%f\n", fdelta, fmin, fmax );
    for( p = 0; p < scn->n; p++ )
      scn->data[ p ] = ( int ) ( 4095.0 * ( data[ p ] - fmin ) / ( fdelta ) ); // 10000 * data[ p ];
    free( data );
  }
  fclose(f);
  return scn;

}







int WriteScene_Analyze(Scene *scn, char filename[])
{
  int len;
  char ext[4];
  char hdr_fname[300], img_fname[300];
  Ana_Hdr1 hdr1;
  Ana_Hdr2 hdr2; 
  int i;
  FILE *f;
  int maxval;
  Voxel v;
  int p,n=0;
  unsigned char val8;
  short int val16;
  long int val32;
  len = strlen(filename);
  ext[0]=tolower(filename[len-3]);
  ext[1]=tolower(filename[len-2]);
  ext[2]=tolower(filename[len-1]);
  ext[3]='\0';
  if (strcmp(ext,"hdr")!=0 && strcmp(ext,"img")!=0) {
    fprintf(stderr,"** File extension has to be \"hdr\" or \"img\"!\n");
    return 1;
  }
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

  
  // WRITING HEADER FILE -----------------
  
  f = fopen(hdr_fname,"wb");
  if (!f) {
    fprintf(stderr,"** unable to open %s for writing.\n", hdr_fname);
    return 1;
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

  ana_fio_abs_write_32(f, 0, 0, hdr1.hdrlen);
  ana_fio_abs_write_zeroes(f, 4, 34);
  ana_fio_abs_write_8(f, 38, hdr1.regular);
  ana_fio_abs_write_8(f, 39, hdr1.hkey0);

  /* -- second header segment -- */

  hdr2.dim[0] = 4;
  hdr2.dim[1] = scn->xsize;
  hdr2.dim[2] = scn->ysize;
  hdr2.dim[3] = scn->zsize;
  hdr2.dim[4] = 1;

  
  maxval = MaximumValue3(scn);
  hdr2.bpp = 8;
  if (maxval>=256) hdr2.bpp = 16;
  if (maxval>=65535) hdr2.bpp = 32;
  hdr2.data_type = hdr2.bpp/4; // == 16) ? 4 : 2;
  hdr2.pixdim[0] = 1.0;
  hdr2.pixdim[1] = scn->dx;
  hdr2.pixdim[2] = scn->dy;
  hdr2.pixdim[3] = scn->dz;
  hdr2.maxval    = maxval;
  hdr2.minval    = 0;

  for(i=0;i<8;i++)
    ana_fio_abs_write_16(f, 40 + 0  + 2*i, 0, hdr2.dim[i]);
  for(i=0;i<7;i++)
    ana_fio_abs_write_16(f, 40 + 16 + 2*i, 0, hdr2.unused[i]);

  ana_fio_abs_write_16(f, 40 + 30, 0, hdr2.data_type);
  ana_fio_abs_write_16(f, 40 + 32, 0, hdr2.bpp);
  ana_fio_abs_write_16(f, 40 + 34, 0, hdr2.dim_un0);

  for(i=0;i<8;i++)
    ana_fio_abs_write_float32(f, 40 + 36 + 4*i, 0, hdr2.pixdim[i]);
  for(i=0;i<8;i++)
    ana_fio_abs_write_float32(f, 40 + 68 + 4*i, 0, hdr2.zeroes[i]);
 
  ana_fio_abs_write_32(f, 40 + 100, 0, hdr2.maxval);
  ana_fio_abs_write_32(f, 40 + 104, 0, hdr2.minval);

  /* -- third header segment (patient info) --- */

  ana_fio_abs_write_zeroes(f, 148, 200);
  fclose(f);


  // WRITING IMAGE FILE ----------------------------

  f = fopen(img_fname, "wb");
  if (!f) {
    fprintf(stderr,"** unable to open %s file.\n",img_fname);
    return 1;
  }
  for (v.z=0; v.z < scn->zsize; v.z++)
    for (v.y=scn->ysize-1; v.y >= 0; v.y--) // reverse order
      for (v.x=0; v.x < scn->xsize; v.x++) {
	p = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (hdr2.bpp==8) {
	  val8 = (unsigned char) scn->data[p];
	  n=fwrite(&val8,sizeof(unsigned char),1,f);
	}
	if (hdr2.bpp==16) {
	  val16 = (short int) scn->data[p];
	  n=fwrite(&val16,sizeof(short int),1,f);
	}
	if (hdr2.bpp==32) {
	  if (hdr2.data_type==8) { // integer 
	    val32 = (long int) scn->data[p];
	    n=fwrite(&val32,sizeof(long int),1,f);
	  } 
	  else {
	    fprintf(stderr,"** Warning: float scene not supported.\n");
	    return 1;
	  }
      	}
	if (n!=1) {
	  fprintf(stderr,"** error reading file %s.\n",img_fname);
	  return 1;
	}
      }
  fclose(f);

  return 0;

}









