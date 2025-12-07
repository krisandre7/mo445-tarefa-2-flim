#include "spectrum.h"
#include "mathematics.h"

int main(int argc, char **argv)
{
  Image *img,*tmp;
  Spectrum *ispec,*dspec,*fspec;
  float Vx,Vy,T;
  //  int p,n;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/
  
  if (argc != 5) {
    fprintf(stderr,"usage: motioncorr <image.pgm> <Vx> <Vy> <T>\n");
    exit(-1);
  }

  img  = ReadImage(argv[1]);
  Vx   = atof(argv[2]);
  Vy   = atof(argv[3]);
  T    = atof(argv[4]);

  ispec = FFT2D(img);
  dspec = MotionDegradation(ispec->ncols,ispec->nrows,Vx,Vy,T);
  fspec = MultSpectrum(ispec,dspec);

  DestroyImage(&img);
  img = ViewMagnitude(dspec);
  WriteImage(img,"filter.pgm");
  DestroyImage(&img);

  img = InvFFT2D(fspec);
  tmp = Abs(img);
  WriteImage(tmp,"restored.pgm");

  DestroyImage(&img);
  DestroyImage(&tmp);
  DestroySpectrum(&ispec);
  DestroySpectrum(&dspec);
  DestroySpectrum(&fspec);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
