#include "ift.h"

/* 
   Author: Alexandre Falcao
   
   Description: This program currently generates a Manhattan plot from
   a simpler p-value file, created by iftGWAA.c
   (PValues_SSS_Simpler.csv), where SSS is a subpopulation. The idea
   is to create also the Q-Q plot from that file in the future. The
   output image, Manhattan.ppm, can be displayed with the display
   command (e.g.: display Manhattan.ppm).

*/

#define ReadBufferSize 4096

/* Data structure to create Manhattan plot and Q-Q plot */

typedef struct ift_snps {
  char         id[12];       // snp identification 
  int          chromosome;   // Chromosome    
  int          position;     // Position TIGR-v6 in the chromossome
  long double  pvalue;       // p-value 
  long double  max_pvalue;   // maximum p-value among the best random ones
  long double  first_pvalue; // first random p-value   
} iftSNPs;

typedef struct ift_geno {
  iftSNPs *snp;
  int      nsnps;   // number of snps
  int      nchroms; // number of chromosomes
  int     *first,*last; // first and last indices of each chromosome
                        // in the snp list
  long double min, max; // minimum and maximum p-values in the snp list
} iftGeno;

iftGeno *iftReadGeno(char *filename)
{
  int      s,nsnps;
  FILE    *fp;
  iftGeno *geno;
  char buffer[ReadBufferSize],*token;

  fp = fopen(filename,"r");  
  fgets(buffer,ReadBufferSize,fp);
  nsnps=0;
  while (!feof(fp)){
    fgets(buffer,ReadBufferSize,fp);
    nsnps++;
  }
  nsnps-=1;
  fclose(fp);

  geno = (iftGeno *)calloc(1,sizeof(iftGeno));
  geno->nsnps = nsnps;
  geno->snp   = (iftSNPs *)calloc(nsnps,sizeof(iftSNPs));

  fp = fopen(filename,"r");  
  fgets(buffer,ReadBufferSize,fp);

  // Read snps
  
  geno->nchroms = 0;
  geno->min     = INFINITY_LDBL;
  geno->max     = INFINITY_LDBL_NEG;
  for (s=0; s < nsnps; s++) {
    fgets(buffer,ReadBufferSize,fp);
    token=strtok(buffer,";");
    strncpy(geno->snp[s].id,token,11); 
    geno->snp[s].id[11]='\0'; 
    token=strtok(NULL,";");
    geno->snp[s].chromosome = atoi(token); 
    token=strtok(NULL,";");
    geno->snp[s].position = atoi(token); 
    token=strtok(NULL,";");
    sscanf(token,"%Le",&geno->snp[s].pvalue); 
    token=strtok(NULL,";");
    sscanf(token,"%Le",&geno->snp[s].max_pvalue); 
    token=strtok(NULL,";");
    sscanf(token,"%Le",&geno->snp[s].first_pvalue); 
    if (geno->snp[s].chromosome > geno->nchroms)
      geno->nchroms = geno->snp[s].chromosome;
    if (geno->snp[s].pvalue < geno->min) 
      geno->min = geno->snp[s].pvalue;
    if (geno->snp[s].pvalue > geno->max) 
      geno->max = geno->snp[s].pvalue;
  }
  
  fclose(fp);

  /* assuming that the positions inside each chromosome are in the
     increasing order */

  geno->first = iftAllocIntArray(geno->nchroms+1);
  geno->last  = iftAllocIntArray(geno->nchroms+1);
  geno->first[geno->snp[0].chromosome] = 0; 
  for (s=0; s < nsnps-1; s++) {
    if (geno->snp[s].chromosome != geno->snp[s+1].chromosome){ 
      geno->last[geno->snp[s].chromosome]    = s;
      geno->first[geno->snp[s+1].chromosome] = s+1;
    }
  }
  geno->last[geno->snp[s].chromosome] = nsnps-1;
  
  return(geno);
}

void iftDestroyGeno(iftGeno *geno)
{
  free(geno->snp);
  free(geno->first);
  free(geno->last);
  free(geno);
}

int main(int argc, char *argv[]) 
{
  iftGeno        *geno=NULL;
  float           window, chrom_size;
  int             s,p,x,y,ymax=INFINITY_INT_NEG;
  iftImage       *img=NULL;
  iftColor         RGB1,YCbCr1,RGB2,YCbCr2;
  iftVoxel         u;
  iftAdjRel       *B=NULL;
  iftColorTable   *ctb=NULL;
  char             command[200];

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=3)
    iftError("Usage: iftGenoPlots <PValues_SSS_Simpler.csv> <output_image.jpg>","main");

  // Read csv file

  geno   = iftReadGeno(argv[1]);

  img = iftCreateImage(2000,(int)(-log10(geno->min)*100)-(int)(-log10(geno->max)*100)+100,1);
  iftSetImage(img,255);
  for(p=0; p < img->n; p++) {
    if (iftGetYCoord(img,p)==(img->ysize-1-400)) // p-value 10 to -4
      img->val[p]=0;
  }
  ctb = iftCreateColorTable(geno->nchroms+1);

  RGB2.val[0] = 0;
  RGB2.val[1] = 0;
  RGB2.val[2] = 0;
  YCbCr2      = iftRGBtoYCbCr(RGB2,255);


  B          = iftCircular(3.0);
  chrom_size = (float)(img->xsize-1)/(float)geno->nchroms;
  printf("chromosome size %f pixels\n",chrom_size);

  window = ROUND(((float)geno->snp[geno->last[geno->snp[0].chromosome]].position - ((float)geno->snp[geno->first[geno->snp[0].chromosome]].position))/chrom_size);
  x = 0;
  y = (int)(-log10(geno->snp[0].pvalue)*100);
  p = x + y*2000;  
  img->val[p]=255;

  RGB1.val[0] = ctb->color[geno->snp[0].chromosome].val[0];
  RGB1.val[1] = ctb->color[geno->snp[0].chromosome].val[1];
  RGB1.val[2] = ctb->color[geno->snp[0].chromosome].val[2];
  YCbCr1      = iftRGBtoYCbCr(RGB1,255);

  for (s=1; s < geno->nsnps; s++) {
    if (geno->snp[s].chromosome != geno->snp[s-1].chromosome){  
      window = ROUND(((float)geno->snp[geno->last[geno->snp[s].chromosome]].position - ((float)geno->snp[geno->first[geno->snp[s].chromosome]].position))/chrom_size);
      RGB1.val[0] = ctb->color[geno->snp[s].chromosome].val[0];
      RGB1.val[1] = ctb->color[geno->snp[s].chromosome].val[1];
      RGB1.val[2] = ctb->color[geno->snp[s].chromosome].val[2];
      YCbCr1      = iftRGBtoYCbCr(RGB1,255);
    }
    x = ((float)(geno->snp[s].position-geno->snp[geno->first[geno->snp[s].chromosome]].position)/window) + chrom_size*(geno->snp[s].chromosome-1);
    y = (int)(-log10(geno->snp[s].pvalue)*100);
    if (y > ymax)      ymax = y;
    if (y > 0){
      p = x + (img->ysize-y-1)*2000;
      u = iftGetVoxelCoord(img,p);
      iftDrawPoint(img,u,YCbCr1,B);
    }
    y = (int)(-log10(geno->snp[s].max_pvalue)*100);
    if (y > ymax)      ymax = y;
    if (y > 0){
      p = x + (img->ysize-y-1)*2000;
      u = iftGetVoxelCoord(img,p);
      iftDrawPoint(img,u,YCbCr2,B);
    }
  }

  if (ymax > 0){
    iftWriteImageP6(img,"temp.ppm");
    sprintf(command,"convert temp.ppm %s",argv[2]);
    system(command);
    sprintf(command,"rm -f temp.ppm");
    system(command);
  }else
    iftWarning("Could not generate plot, because all p-values are 1.0","main");

  iftDestroyImage(&img);
  iftDestroyGeno(geno);
  iftDestroyAdjRel(&B);
  iftDestroyColorTable(&ctb);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



