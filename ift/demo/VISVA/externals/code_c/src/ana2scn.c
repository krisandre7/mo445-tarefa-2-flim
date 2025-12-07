/*
   Analyze -> SCN conversion

   (C) 2003 Felipe P.G. Bergo <bergo@seul.org>
   2003.11.03
*/

#include "ana2scn.h"


struct _hdr {

  int hdrlen;
  int bpp;
  int dimensions;
  int W,H,D,T;
  float dx,dy,dz;
  int be_hint;
  int dt;

} hdr;

void fgoto_0(int x,int y,int z,FILE *f, int bpp, long poff);
void read_img(char *hdr_name, char *out_name, int obpp);
void read_hdr(char *hdr_name, int obpp);


void fgoto_0(int x,int y,int z,FILE *f, int bpp, long poff) {
  long off;
  off = (bpp/8) * (x + (hdr.W*(hdr.H-1-y)) + (hdr.W * hdr.H) * z);
  fseek(f,poff+off,SEEK_SET);  
}

void read_img(char *hdr_name, char *out_name, int obpp){
  FILE *f, *g;
  char *img_name;
  int s,i,j,k,r;
  unsigned char *buf8;
  short int *buf16;
  int *buf32;
  float *bufx;
  int vmax,nbits;
  float fmax;
  long poff;
  
  s = strlen(hdr_name);
  img_name = (char *) malloc(s + 1);
  strcpy(img_name, hdr_name);

  strcpy(img_name + s - 3, "img");
  f = fopen(img_name, "r");
  if (!f) {
    strcpy(img_name + s - 3, "IMG");
    f = fopen(img_name, "r");
  }
  if (!f) {
    strcpy(img_name + s - 3, "Img");
    f = fopen(img_name, "r");
  }

  if (!f) {
    fprintf(stderr,"** unable to open .img file.\n");
    exit(2);
  }
  
  g = fopen(out_name, "w");
  if (!g) {
    fprintf(stderr,"** unable to open %s for writing.\n",out_name);
    exit(2);
  }

  fprintf(g,"SCN\n%d %d %d\n%.4f %.4f %.4f\n%d\n",
	  hdr.W, hdr.H, hdr.D, hdr.dx, hdr.dy, hdr.dz, obpp);
  poff = ftell(g);

  j = hdr.H * hdr.D;

  switch(hdr.bpp) {

  case 8:
    buf8 = (unsigned char *) malloc(hdr.W * 4);
    buf16 = (short int *) buf8;
    buf32 = (int *) buf8;
    if (!buf8) {
      fprintf(stderr,"** malloc error.\n");
      exit(3);
    }
    for(i=0;i<j;i++) {
      if (fread(buf8, 1, hdr.W, f) != hdr.W) {
	fprintf(stderr,"** read error.\n");
	exit(4);
      }

      fgoto_0(0,i%(hdr.H),i/(hdr.H), g, obpp, poff);

      switch(obpp) {
      case 8: 
	r = fwrite(buf8, 1, hdr.W, g); break;
      case 16:
	for(k=hdr.W-1;k>=0;k--) buf16[k] = (short int) buf8[k];
	r = fwrite(buf8, 2, hdr.W, g); break;
      case 32:
	for(k=hdr.W-1;k>=0;k--) buf32[k] = (int) buf8[k];
	r = fwrite(buf8, 4, hdr.W, g); break;
      default: r = -1;
      }
      
      if (r != hdr.W) {
	fprintf(stderr,"** write error.\n");
	exit(4);
      }   
    }
    free(buf8);
    break;

  case 16:
    buf8 = (unsigned char *) malloc(hdr.W * 4);
    buf16 = (short int *) buf8;
    buf32 = (int *) buf8;
    if (!buf8) {
      fprintf(stderr,"** malloc error.\n");
      exit(3);
    }
    for(i=0;i<j;i++) {
      if (fread(buf8, 2, hdr.W, f) != hdr.W) {
	fprintf(stderr,"** read error.\n");
	exit(4);
      }
      /* turn values to little-endian if needed */
      if (hdr.be_hint)
	for(k=0;k<hdr.W;k++)
	  buf16[k] = (short int) fio_swab_16(buf16[k]);

      fgoto_0(0,i%(hdr.H),i/(hdr.H), g, obpp, poff);

      switch(obpp) {
      case 16: r = fwrite(buf8, 2, hdr.W, g); break;
      case 8:
	for(k=0;k<hdr.W;k++) buf8[k] = (unsigned char) (buf16[k]>>8);
	r = fwrite(buf8, 1, hdr.W, g); break;
      case 32:
	for(k=hdr.W-1;k>=0;k--) buf32[k] = (int) buf16[k];
	r = fwrite(buf8, 4, hdr.W, g); break;
      default: r = -1;
      }

      if (r != hdr.W) {
	fprintf(stderr,"** write error.\n");
	exit(4);
      }   
    }
    free(buf8);
    break;

  case 32:
    buf8 = (unsigned char *) malloc(hdr.W * 4);
    buf16 = (short int *) buf8;
    buf32 = (int *) buf8;
    bufx = (float *) buf8;
    if (!buf8) {
      fprintf(stderr,"** malloc error.\n");
      exit(3);
    }

    if (hdr.dt == 8) { // integer data
      vmax = 0;
      for(i=0;i<j;i++) {
	if (fread(buf8, 4, hdr.W, f) != hdr.W) {
	  fprintf(stderr,"** read error.\n");
	  exit(4);
	}
	if (hdr.be_hint)
	  for(k=0;k<hdr.W;k++)
	    buf32[k] = (int) fio_swab_32(buf32[k]);
	for(k=0;k<hdr.W;k++) if (buf32[k]>vmax) vmax = buf32[k];
      }
      fseek(f,0,SEEK_SET);
      nbits = 0; while(vmax>0) { nbits++; vmax >>= 1; }
      
      for(i=0;i<j;i++) {
	if (fread(buf8, 4, hdr.W, f) != hdr.W) {
	  fprintf(stderr,"** read error.\n");
	  exit(4);
	}
	
	/* turn values to little-endian if needed */
	if (hdr.be_hint)
	  for(k=0;k<hdr.W;k++)
	    buf32[k] = (int) fio_swab_32(buf32[k]);
	
	fgoto_0(0,i%(hdr.H),i/(hdr.H), g, obpp, poff);
	switch(obpp) {
	case 32: r = fwrite(buf8, 4, hdr.W, g); break;
	case 8:
	  for(k=0;k<hdr.W;k++) buf8[k] = (unsigned char) (buf32[k]>>(nbits-8));
	  r = fwrite(buf8, 1, hdr.W, g); break;
	case 16:
	  for(k=0;k<hdr.W;k++) buf16[k] = (short int) (buf32[k]>>(nbits-15));
	  r = fwrite(buf8, 2, hdr.W, g); break;
	default: r = -1;
	}

	if (r != hdr.W) {
	  fprintf(stderr,"** write error.\n");
	  exit(4);
	}   
      }
    } else if (hdr.dt==16) {
      fprintf(stderr,"warning: converting floating-point data to integer.\n");
      // float data
      fmax = 0.0;
      for(i=0;i<j;i++) {
	if (fread(buf8, 4, hdr.W, f) != hdr.W) {
	  fprintf(stderr,"** read error.\n");
	  exit(4);
	}
	if (hdr.be_hint)
	  for(k=0;k<hdr.W;k++)
	    buf32[k] = (int) fio_swab_32(buf32[k]);
	for(k=0;k<hdr.W;k++) if (bufx[k]>fmax) fmax = bufx[k];
      }
      fseek(f,0,SEEK_SET);

      for(i=0;i<j;i++) {
	if (fread(buf8, 4, hdr.W, f) != hdr.W) {
	  fprintf(stderr,"** read error.\n");
	  exit(4);
	}
	
	/* turn values to little-endian if needed */
	if (hdr.be_hint)
	  for(k=0;k<hdr.W;k++)
	    buf32[k] = (int) fio_swab_32(buf32[k]);

	fgoto_0(0,i%(hdr.H),i/(hdr.H), g, obpp, poff);

	switch(obpp) {
	case 32: 
	  if (fmax > 2147483647.0)
	    for(k=0;k<hdr.W;k++) buf32[k] = (int) (bufx[k] * (2147483647.0 / fmax));

	  else
	    for(k=0;k<hdr.W;k++) buf32[k] = (int) bufx[k];
	  r = fwrite(buf8, 4, hdr.W, g); break;
	case 8:
	  if (fmax > 255.0)
	    for(k=0;k<hdr.W;k++) buf8[k] = (unsigned char) (bufx[k] * (255.0 / fmax));

	  else
	    for(k=0;k<hdr.W;k++) buf8[k] = (unsigned char) bufx[k];
	  r = fwrite(buf8, 1, hdr.W, g); break;
	case 16:
	  if (fmax > 32767.0)
	    for(k=0;k<hdr.W;k++) buf16[k] = (short int) (bufx[k] * (32767.0 / fmax));

	  else
	    for(k=0;k<hdr.W;k++) buf16[k] = (short int) bufx[k];
	  r = fwrite(buf8, 2, hdr.W, g); break;
	default: r = -1;
	}

	if (r != hdr.W) {
	  fprintf(stderr,"** write error.\n");
	  exit(4);
	}   
      }

    } else {
      fprintf(stderr,"** unknown datatype %d.\n",hdr.dt);
      exit(4);
    }
    free(buf8);
    break;


  }
  free(img_name);
  fclose(f);
  fclose(g);
}

void read_hdr(char *hdr_name, int obpp){
  FILE *f;
  int be = 0;

  f = fopen(hdr_name,"r");
  if (!f) {
    fprintf(stderr,"** unable to open %s for reading.\n", hdr_name);
    exit(2);
  }

 again:
  hdr.hdrlen = fio_abs_read_32(f,0,be);
  if (hdr.hdrlen != 348 && be==0) {
    be = !be;
    goto again;
  }

  if (hdr.hdrlen!=348) {
    fprintf(stderr,"** This is not an Analyze header.\n");
    exit(2);
  }

  hdr.dt           = fio_abs_read_16(f,40+30,be);
  hdr.bpp          = fio_abs_read_16(f,40+32,be);
  hdr.dimensions   = fio_abs_read_16(f,40,be);
  hdr.W            = fio_abs_read_16(f,40+2,be);
  hdr.H            = fio_abs_read_16(f,40+4,be);
  hdr.D            = fio_abs_read_16(f,40+6,be);
  hdr.T            = fio_abs_read_16(f,40+8,be);

  hdr.dx           = fio_abs_read_float32(f,40+36+4,be);
  hdr.dy           = fio_abs_read_float32(f,40+36+8,be);
  hdr.dz           = fio_abs_read_float32(f,40+36+12,be);

  fclose(f);

  if (hdr.dimensions == 4 && hdr.T != 1) {
    fprintf(stderr,"** This file has a 3D time series, I can't convert it.\n");
    exit(2);
  }

  if (hdr.dimensions < 3 || hdr.dimensions > 4) {
    fprintf(stderr,"** This file is not a 3D volume, I can't convert it.\n");
    exit(2);
  }
  if (obpp < 0) obpp = hdr.bpp;

  if (hdr.bpp != 8 && hdr.bpp != 16 && hdr.bpp != 32) {
    fprintf(stderr,"** This is %d-bit data, only 8-, 16-bit and 32-bit data are supported.\n",hdr.bpp);
    exit(2);
  }

  hdr.be_hint = be;

}



void read_analyze(char *hdr_name, char *out_name, int obpp){
  read_hdr(hdr_name, obpp);
  read_img(hdr_name, out_name, obpp);
}

