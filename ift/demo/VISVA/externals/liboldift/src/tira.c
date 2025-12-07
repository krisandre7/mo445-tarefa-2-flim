/* Texture Image Retrieval Algorithm */

#include <string.h> /* memset() */
#include <stdlib.h>
#include <math.h> /* sqrt() */

#include "tira.h"
#include "dimage.h"
#include "common.h"
#include "radiometric.h"


/** Colocamos aqui porque sempre deslocamos os filtros! **/
/* 1=desloca os filtros projetados centralizados, para preparar pra multiplicacao */
#define DESLOCA 1


/******************************************************************************
 * Cabeçalho das Funções Internas
 ******************************************************************************/
void _TiraEscala( Image *img, int scale, int nrotations, Vc *ret_vect );
void _TiraEscalaRot( Image *img, int nrotations, int rotation, Vc *ret_vect );
Image *_FiltraEscalaRot(Image *img, int nrotations, int rotation);

Image *_FilterImage(Image *img, Spectrum *flt);
Image *_FilterImageLowPass(Image *img);
Spectrum *_CriaFiltroPassaBaixas(Image *img);
Spectrum *_CriaFiltroPassaBaixas2(Image *img);
Spectrum *_FltDirecional(Image *img, double theta, int nrotations, int rotation);
void _NormalizeSpectrum(Spectrum *spec);

Image *_DownSample2(Image *img);
void _MergeVc(Vc *vaux, Vc *vect, int bin, int nrotations);

double _GetMu(Image *img);
double _GetSigma(Image *img);

void _AlinhaOrientacaoDescritor(Vc *vect);
void _AlinhaEscalaDescritor(Vc *vect);
void _AlinhaDescritor(Vc *vect);

double _fact(double n);
double _FindAngle(int x1, int y1, int x2, int y2);

void _Transf2Positive(Vc *vect);
double _GetValMinNegative(Vc *vect);

/******************************************************************************
 * Cabeçalho do filtro direcional que é uma reta
 ******************************************************************************/
bool _SValidPixel(Spectrum *flt, int x, int y);
void _DrawSpectrumLineDDA(Spectrum *flt, Spectrum *spec, int x1, int y1, int xn, int yn);
Spectrum *_FltDirecionalRETA(Image *img, double theta, int nrotations, int rotation);

/******************************************************************************
 * Cabeçalho das Funções Internas de Vetor Caracteristico direto do espaco de
 * frequencias
 ******************************************************************************/
Spectrum *_SFiltraEscalaRot(Image *img, int nrotations, int rotation); /**/
Spectrum *_SFilterImage(Image *img, Spectrum *flt); /**/
double _SGetMu(Spectrum *img); /**/
double _SGetSigma(Spectrum *img); /**/

char *get_extension(char *fname);

/******************************************************************************
 * Funções Externas
 ******************************************************************************/

/* rotinas para gerenciamento de Vetor de Características */
Vc *CreateVc(int nscales, int nrotations)
{
  Vc *vect=NULL;

  vect = (Vc *) calloc(1,sizeof(Vc));
  if (vect == NULL){
    Error(MSG1,"CreateVc");
  }

  vect->mu    = AllocDoubleArray(nscales*nrotations);
  vect->sigma = AllocDoubleArray(nscales*nrotations);

  vect->nscales     = nscales;
  vect->nrotations  = nrotations;

  return(vect);
}

void DestroyVc( Vc **vect)
{
  Vc *aux;

  aux = *vect;
  if(aux != NULL){
    if (aux->mu != NULL)    free(aux->mu); 
    if (aux->sigma != NULL) free(aux->sigma);
    free(aux);
    *vect = NULL;
  }
}


void PrintVc(Vc *vect)
{
  register int s, k;

  if (vect == NULL) {
    fprintf(stdout, "PrintVc: NULL Param\n");
    return;
  }

  fprintf(stdout, "Energy:\n");
  for (s=0; s < vect->nscales; s++){
    fprintf(stdout, "\tScale: %3d\t",s);
    for (k=0; k < vect->nrotations; k++){
      fprintf(stdout, "%5.2f ", vect->mu[(s*vect->nrotations)+k]);
    }
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "\n");

  fprintf(stdout, "Deviation:\n");
  for (s=0; s < vect->nscales; s++){
    fprintf(stdout, "\tScale: %3d\t",s);
    for (k=0; k < vect->nrotations; k++){
      fprintf(stdout, "%5.2f ", vect->sigma[(s*vect->nrotations)+k]);
    }
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "\n");
}


/* extração do vetor de caracteristicas da textura de imagem */
Vc *Tira( Image *img, int nscales, int nrotations )
{
  Vc *vect, *vaux;
  Image *iaux, *itmp, *iaux2;                    /* imagem auxiliar */
  int x, y, s;  

  vect = CreateVc(nscales, nrotations); /* vetor final retornado */
  vaux = CreateVc(1, nrotations);       /* vetor auxiliar, usado uma vez para
                                         * cada uma das escalas */
#if SAIDA_RELATORIO
  WriteImage(img, "img_l0.pgm");
#endif

  /* filtra a imagem com um passa baixa */
  itmp = _FilterImageLowPass(img);
  iaux = CreateImage(img->ncols, img->nrows);
  for(x=0; x < iaux->nrows; x++) {
    for(y=0; y < iaux->ncols; y++) {
      iaux->val[iaux->tbrow[x]+y] = itmp->val[itmp->tbrow[x]+y];
    }
  }
  DestroyImage(&itmp);
  itmp = NULL;

#if SAIDA_RELATORIO
  WriteImage(iaux, "img_l1.pgm");
#endif


  for (s=0; s < nscales; s++){
    /* cria o `sub-vetor' caracteristico para aquela escala, dada uma imagem,
     * obtem também a imagem filtrada com um passa baixas */
    _TiraEscala(iaux, s, nrotations, vaux);
    
    /* faz o merge do `sub-vetor' caracteristico da escala s dentro do vetor de
     * resposta */
    _MergeVc(vaux, vect, s, nrotations);

    /* faz o down-sampling da imagem filtrada `iaux' e a devolve em `iaux',
     * a não ser que seja o caso da última escala, no qual pulamos essa parte */
    if (s < nscales-1 ) {
#if SAIDA_RELATORIO
  WriteImage(iaux, "img_l2.pgm");
#endif
      /* Aplica um passa baixa na imagem de entrada */

      itmp = _FilterImageLowPass(iaux);
#if SAIDA_RELATORIO
  WriteImage(itmp, "img_tmp.pgm");
#endif
      iaux2 = CreateImage(iaux->ncols, iaux->nrows);
      for(x=0; x < iaux->nrows; x++) {
        for(y=0; y < iaux->ncols; y++) {
          iaux2->val[iaux->tbrow[x]+y] = itmp->val[itmp->tbrow[x]+y];
        }
      }
      DestroyImage(&itmp);
      DestroyImage(&iaux);
      iaux = NULL;

      /* aplica downsampling na imagem */
      iaux = _DownSample2(iaux2);
      DestroyImage(&iaux2);
      iaux2 = NULL;

#if SAIDA_RELATORIO
  WriteImage(iaux, "img_l3.pgm");
#endif

    }
  }

  /* libera a memória das variaveis auxiliares */
  DestroyVc(&vaux);
  DestroyImage(&iaux);

  return vect;
}


/******************************************************************************
 * Funções Internas
 ******************************************************************************/

void _TiraEscala( Image *img, int scale, int nrotations, Vc *ret_vect )
{
  int k;
  Vc *vaux;

  vaux = CreateVc(1, 1);       /* vetor auxiliar, usado uma vez para
                                * cada uma das escalas */

  for (k=0; k < nrotations; k++){
    /* cria o `sub-vetor' caracteristico para aquela escala, para cada rotacao */
    _TiraEscalaRot( img, nrotations, k, vaux);
    
    /* faz o merge do `sub-vetor' caracteristico da escala s dentro do vetor de
     * resposta */
    _MergeVc(vaux, ret_vect, k, 1);

  }

  /* libera memória das variáveis auxiliares */
  DestroyVc (&vaux);
}


void _TiraEscalaRot( Image *img, int nrotations, int rotation, Vc *ret_vect )
{
#if FEATURES_NO_ESPACO
  int x, y;

  Image *itmp;
  Image *iaux;

  /* Aplica um filtro direcional na imagem img */
  itmp = _FiltraEscalaRot(img, nrotations, rotation);

  /* Voltando com a imagem nas dimensoes reais */
  iaux = CreateImage(img->ncols, img->nrows);
  for(x=0; x < img->nrows; x++) {
    for(y=0; y < img->ncols; y++) {
      iaux->val[iaux->tbrow[x]+y] = itmp->val[itmp->tbrow[x]+y];
    }
  }
  DestroyImage(&itmp);
  itmp = NULL;

  /* Extrai os parâmetros da imagem */
  ret_vect->mu[0] = _GetMu(iaux);
  ret_vect->sigma[0] = _GetSigma(iaux);

  /* libera memória das variáveis auxiliares */
  DestroyImage(&iaux);
#else
  Spectrum *faux;

  /* Aplica um filtro direcional na imagem img */
  faux = _SFiltraEscalaRot(img, nrotations, rotation);

  /* Extrai os parâmetros da imagem */
  ret_vect->mu[0] = _SGetMu(faux);
  ret_vect->sigma[0] = _SGetSigma(faux);

  /* libera memória das variáveis auxiliares */
  DestroySpectrum(&faux);
#endif

}


Image *_FiltraEscalaRot(Image *img, int nrotations, int rotation)
{
  Spectrum *flt;
  Image *iresult;

  /* cria filtro direcional */
#if ESCOLHE_DIRECIONAL
  flt = _FltDirecional(img, ((float)rotation)*M_PI/((float) nrotations), nrotations, rotation);
#else
  flt = _FltDirecionalRETA(img, ((float)rotation)*M_PI/((float) nrotations), nrotations, rotation);
#endif

#if SAIDA_RELATORIO
  {
  Image *ig;
  ig = ViewMagnitude(flt);
  if (rotation==0) WriteImage(ig, "dirfilter0.pgm");
  if (rotation==1) WriteImage(ig, "dirfilter1.pgm");
  if (rotation==2) WriteImage(ig, "dirfilter2.pgm");
  if (rotation==3) WriteImage(ig, "dirfilter3.pgm");
  if (rotation==4) WriteImage(ig, "dirfilter4.pgm");
  if (rotation==5) WriteImage(ig, "dirfilter5.pgm");
  if (rotation==6) WriteImage(ig, "dirfilter6.pgm");
  DestroyImage(&ig);
  }
#endif


  /* guarda imagem para retornar */
  iresult = _FilterImage(img, flt);

#if SAIDA_RELATORIO
  if (rotation==0) WriteImage(iresult, "idirfilter0.pgm");
  if (rotation==1) WriteImage(iresult, "idirfilter1.pgm");
  if (rotation==2) WriteImage(iresult, "idirfilter2.pgm");
  if (rotation==3) WriteImage(iresult, "idirfilter3.pgm");
  if (rotation==4) WriteImage(iresult, "idirfilter4.pgm");
  if (rotation==5) WriteImage(iresult, "idirfilter5.pgm");
  if (rotation==6) WriteImage(iresult, "idirfilter6.pgm");
#endif

  /* apaga memória auxiliar */
  DestroySpectrum(&flt);

  return iresult;
}





Image *_FilterImage(Image *img, Spectrum *flt)
{
  Spectrum *fimg, *fresult;
  Image *iresult;

#if USE_DIMAGE
  DImage *diaux;
#if USE_IMAGE_ADD_MIN
    int p;
    int imin;
#endif
#endif

  /* imagem para o dominio da frequencia */
  fimg = FFT2D(img);

  /* zerando a parte imaginária da imagem */
  //memset( fimg->imag, '\0', sizeof(double)*fimg->ncols*fimg->nrows );

  /* imagem filtrada com flt, no dominio da frequencia */
  fresult = MultSpectrum(fimg,flt);
  /* imagem filtrada no dominio do espaco */
#if USE_DIMAGE
  diaux = DInvFFT2D(fresult);
  iresult = ConvertDImage2Image(diaux);
  DestroyDImage(&diaux);
#else
  iresult = InvFFT2D(fresult);
#if USE_IMAGE_ADD_MIN
  imin = MinimumValue(iresult);
  for (p=0; p<iresult->nrows*iresult->ncols; p++)
    iresult->val[p] += imin;
#endif
#endif
  /* apaga auxiliares */
  DestroySpectrum(&fimg);
  DestroySpectrum(&fresult);

  /* retorna imagem filtrada no dominio do espaco */
  return iresult;
}


Image *_FilterImageLowPass(Image *img)
{
  Spectrum *flt;
  Image *iresult;

#if SAIDA_RELATORIO
  WriteImage(img, "lp_img0.pgm");
#endif
  /* cria o filtro passa baixas do tamanho da imagem */
#if ESCOLHE_PASSA_BAIXAS
  flt = _CriaFiltroPassaBaixas(img);
#else
  flt = _CriaFiltroPassaBaixas2(img);
#endif
  /* guarda imagem para retornar */
  iresult = _FilterImage(img, flt);

#if SAIDA_RELATORIO
  WriteImage(img, "lp_img1.pgm");
  WriteImage(iresult, "lp_iresult.pgm");
#endif

  /* apaga memória auxiliar */
  DestroySpectrum(&flt);

  return iresult;
}


Spectrum *_CriaFiltroPassaBaixas(Image *img)
{
  Spectrum *spec;
  int ncols, nrows, x, y, p, pb;
  double r, fc;

  /* Enlarge ncols and nrows to the nearest power of 2 */
  ncols = (int)ceil(log(img->ncols)/log(2));
  nrows = (int)ceil(log(img->nrows)/log(2));
  ncols = 1 << ncols;
  nrows = 1 << nrows;

  spec = CreateSpectrum(ncols, nrows);
  fc = MIN(img->ncols,img->ncols)/8;

  /* trazendo o filtro pro tamanho do dominio da imagem */
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      r = (int) sqrt( (x-nrows/2)*(x-nrows/2) + (y-ncols/2)*(y-ncols/2) );
      spec->imag[p] = 1.0/(1+0.414*pow(r/fc,2.0));
    }
  }

  /* usando a parte imaginaria para criar o filtro centralizado */
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      r = (int) sqrt( (x-nrows/2)*(x-nrows/2) + (y-ncols/2)*(y-ncols/2) );
      spec->imag[p] = 1.0/(1+0.414*pow(r/fc,2.0));
    }
  }


#if DESLOCA
  /* deslocando o filtro para o lugar certo, na parte real*/
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      pb = ((x+nrows/2)%nrows)*ncols + (y+ncols/2)%ncols;
      spec->real[p] = spec->imag[pb];
    }
  }
#else
  /* deslocando o filtro para o lugar certo, na parte real*/
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      pb = x*ncols+y;
      spec->real[p] = spec->imag[pb];
    }
  }
#endif

  /* zerando a parte imaginária */
  memset( spec->imag, '\0', sizeof(double)*ncols*nrows );

  _NormalizeSpectrum(spec);


#if SAIDA_RELATORIO
  { /* testando o filtro gerado */
    Image *test;
    test = ViewMagnitude(spec);
    WriteImage(test,"filtro_passa_baixas.pgm");
    DestroyImage(&test);
  }
#endif

  /* retorna o filtro passa baixas */
  return spec;

}


Spectrum *_CriaFiltroPassaBaixas2(Image *img)
{
  Spectrum *spec;
  int ncols, nrows, x, y, p, pb;
  double r, fc;

  /* Enlarge ncols and nrows to the nearest power of 2 */
  ncols = (int)ceil(log(img->ncols)/log(2));
  nrows = (int)ceil(log(img->nrows)/log(2));
  ncols = 1 << ncols;
  nrows = 1 << nrows;

  spec = CreateSpectrum(ncols, nrows);
  fc = MIN(img->ncols,img->ncols)/2;

  /* usando a parte imaginaria para criar o filtro centralizado */
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      r = (int) sqrt( (x-nrows/2)*(x-nrows/2) + (y-ncols/2)*(y-ncols/2) );
      if (r <= fc/4.0) {
        spec->imag[p] = 1.0;
      } else if (r < fc/2.0) {
        spec->imag[p] = cos(M_PI_2 * log10(4.0*r/fc)/log10(2.0) );
      } else {
        spec->imag[p] = 0.0;
      }
    }
  }


#if DESLOCA
  /* deslocando o filtro para o lugar certo, na parte real*/
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      pb = ((x+nrows/2)%nrows)*ncols + (y+ncols/2)%ncols;
      spec->real[p] = spec->imag[pb];
    }
  }
#else
  /* deslocando o filtro para o lugar certo, na parte real*/
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      pb = x*ncols+y;
      spec->real[p] = spec->imag[pb];
    }
  }
#endif

  /* zerando a parte imaginária */
  memset( spec->imag, '\0', sizeof(double)*ncols*nrows );

  _NormalizeSpectrum(spec);


#if SAIDA_RELATORIO
  { /* testando o filtro gerado */
    Image *test;
    test = ViewMagnitude(spec);
    WriteImage(test,"filtro_passa_baixas.pgm");
    DestroyImage(&test);
  }
#endif

  /* retorna o filtro passa baixas */
  return spec;

}


Spectrum *_FltDirecional(Image *img, double theta, int nrotations, int rotation)
{
  Spectrum *spec;
  int ncols, nrows, x, y, p, pb;
  double r, fc, ak, angle;


  /* Enlarge ncols and nrows to the nearest power of 2 */
  ncols = (int)ceil(log(img->ncols)/log(2));
  nrows = (int)ceil(log(img->nrows)/log(2));
  ncols = 1 << ncols;
  nrows = 1 << nrows;

  spec = CreateSpectrum(ncols, nrows);
  fc = MIN(img->nrows,img->ncols)/2;
  //ak = pow(2.0,nrotations-1) * _fact(nrotations-1) / sqrt(nrotations*_fact(2*(nrotations-1)));
  ak = 1.0;
  

  /* usando a parte imaginaria para criar o filtro centralizado */
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {

      //angle = _FindAngle(1.0, 0.0,(y-ncols/2),(x-nrows/2)) - theta - M_PI;
      angle = atan2((y-ncols/2),(x-nrows/2)) + M_PI - theta;
      p = x*ncols+y;
      r = (int) sqrt( (x-nrows/2)*(x-nrows/2) + (y-ncols/2)*(y-ncols/2) );

#if 0
      /* passa baixas v2*/
      if (r <= fc/2.0) {
        spec->imag[p] = 1.0;
      } else if (r < 3*fc/4.0) {
        spec->imag[p] = cos(M_PI_2 * log10(2.0*r/fc)/log10(2.0) );
      } else { /* r >= 3*fc/4.0 */
        spec->imag[p] = 0.0;
      }
#endif
#if LIGA_PASSA_BAIXA_DIRECIONAL
      /* passa baixas v1 */
      if (r <= fc/4.0) {
        spec->imag[p] = 1.0;
      } else if (r < fc/2.0) {
        spec->imag[p] = cos(M_PI_2 * log10(4.0*r/fc)/log10(2.0) );
      } else { /* r >= fc/2.0 */
        spec->imag[p] = 0.0;
      }
#else
      spec->imag[p] = 1.0;
#endif

      /* passa altas */
      if (r <= fc/4.0) {
        spec->imag[p] = 0.0;
      } else if ((r < fc/2.0) && (r > fc/4.0)) {
        spec->imag[p] *= cos(M_PI_2 * log10(2.0*r/fc)/log10(2.0) );
      } else spec->imag[p] *= 1.0;

#if 1

      /* direcional v2 */
      if (fabs(angle) < M_PI/(nrotations*2) || fabs(2*M_PI - angle) < M_PI/(nrotations*2) ||
          fabs(M_PI - angle) < M_PI/(nrotations*2) || fabs(angle + M_PI) < M_PI/(nrotations*2) ) {
        spec->imag[p] *= ak*pow(cos(angle),nrotations-1);
      } else {
        spec->imag[p] = 0.0;
      }
#endif
#if 0
      /* direcional v1 */
      if (fabs(angle) < M_PI_2 || fabs(2*M_PI - angle) < M_PI_2 ||
          fabs(M_PI - angle) < M_PI_2 || fabs(angle + M_PI) < M_PI_2 ) {
        spec->imag[p] *= ak*pow(cos(angle),nrotations-1);
      } else {
        spec->imag[p] = 0.0;
      }
#endif

    }
  }

#if DESLOCA
  /* deslocando o filtro para o lugar certo, na parte real*/
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      pb = ((x+nrows/2)%nrows)*ncols + (y+ncols/2)%ncols;
      spec->real[p] = spec->imag[pb];
    }
  }
#else
  /* deslocando o filtro para o lugar certo, na parte real*/
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      pb = x*ncols+y;
      spec->real[p] = spec->imag[pb];
    }
  }
#endif

  /* zerando a parte imaginária */
  memset( spec->imag, '\0', sizeof(double)*ncols*nrows );

  _NormalizeSpectrum(spec);


  /* retorna o filtro gerado */
  return spec;

}

void _NormalizeSpectrum(Spectrum *spec)
{
  double max = 1e-15;

  int i, j, k;
  for (i=0; i< spec->nrows; i++) {
    for (j=0; j< spec->ncols; j++) {
      k = spec->tbrow[i]+j;
      if (spec->real[k] > max)
        max = spec->real[k];
    }
  }

  /* Normalize the image to [0, 1] interval  */
  for (i=0; i< (spec->nrows*spec->ncols); i++)
    spec->real[i] = fabs(spec->real[i]/max);
}





Image *_DownSample2(Image *img)
{
  int c, r, ncols, nrows;
  Image *ret_img;

  /* calcula novas dimensoes */
  ncols = (int) (img->ncols/2 + 0.5);
  nrows = (int) (img->nrows/2 + 0.5);

  /* cria a imagem nova */
  ret_img = CreateImage(ncols, nrows);

  /* copia um pixel a cada dois */
  for (r=0; r<nrows; r++) {
    for (c=0; c<ncols; c++) {
      ret_img->val[(r*ncols) + c] = img->val[(img->tbrow[r*2]) + 2*c];
    }
  }

  /* retorna a imagem aplicando down sample*/
  return ret_img;

}


void _MergeVc(Vc *vaux, Vc *vect, int s, int nrotations)
{
  int k;

  for (k=0; k < nrotations; k++) {
    vect->mu[s*nrotations + k] = vaux->mu[k];
    vect->sigma[s*nrotations + k] = vaux->sigma[k];
  }
}





/*	Retorna a energia média acumulada da imagem. */
double _GetMu(Image *img) {
  int i, n;
  double energia_acumulada = 0.0;
  n = img->ncols * img->nrows;
  
// calcula a energia acumulada... 
  for (i = 0; i < n; i++) {
#if MU_QUAD
    energia_acumulada += pow(img->val[i],2);
#else
    energia_acumulada += fabs(img->val[i]);
#endif
  }
  return (energia_acumulada/n);
}

/* Retorna o desvio padrão da energia da imagem. */
double _GetSigma(Image *img) {
  int i, n;
  double media = _GetMu(img), desvio = 0.0;
  
  n= img->ncols*img->nrows;
  for (i = 0; i < n; i++) {
    desvio =+ pow(img->val[i] - media, 2);
  }
  return (sqrt(desvio/n));
}





/* Realiza alinhamento na orientação do descritor. */
void _AlinhaOrientacaoDescritor(Vc * vect) {
  Vc *vect_aux;
	double mu_ac, mu_ac_max;
	int orientacao_dominante, k, s, x, i;

  vect_aux = CreateVc(vect->nscales, vect->nrotations);

  orientacao_dominante = 0;
  mu_ac_max = 0.0;

  // Encontrando a orientação dominante ao longo das escalas...
	for (k=0; k < vect->nrotations; k++){
    mu_ac = 0.0;
  	for (s=0; s < vect->nscales; s++){
      mu_ac += vect->mu[(s*vect->nrotations)+k];
    }
		
		if (mu_ac > mu_ac_max) {
			mu_ac_max = mu_ac;
			orientacao_dominante = k;
		}
  }

  // Gravando os valores de orientações corrigidos no vetor de características de saída...
  for (s = 0; s < vect->nscales; s++){
    for (i = 0, x = orientacao_dominante; x < vect->nrotations; x++, i++) {
		  vect_aux->mu[(s*vect->nrotations) + i] =  vect->mu[(s*vect->nrotations) + x];
			vect_aux->sigma[(s*vect->nrotations) + i] =  vect->sigma[(s*vect->nrotations)+x];
    }

    for (x = 0, i = vect->nrotations - orientacao_dominante; x < orientacao_dominante; i++, x++) {
		  vect_aux->mu[(s*vect->nrotations) + i] =  vect->mu[(s*vect->nrotations)+x];
			vect_aux->sigma[(s*vect->nrotations) + i] =  vect->sigma[(s*vect->nrotations)+x];
    }
  }

  for (x = 0; x < (vect->nscales * vect->nrotations); x++) {
    vect->mu[x] =  vect_aux->mu[x];
		vect->sigma[x] =  vect_aux->sigma[x];
  }
  DestroyVc(&vect_aux);
}

/* Realiza alinhamento na escala do descritor. */
void _AlinhaEscalaDescritor(Vc * vect) {
  Vc *vect_aux;  
	double mu_ac, mu_ac_max;
	int escala_dominante, k, s, x, i;

  vect_aux = CreateVc(vect->nscales, vect->nrotations);

  escala_dominante = 0;
  mu_ac_max = 0.0;

  // Encontrando a escala dominante ao longo das orientações...
  for (s=0; s < vect->nscales; s++){
    mu_ac = 0.0;
    for (k=0; k < vect->nrotations; k++){
      mu_ac += vect->mu[(s*vect->nrotations)+k];
    }

		if (mu_ac > mu_ac_max) {
			mu_ac_max = mu_ac;
			escala_dominante = s;
		}
  }

  
  // Gravando os valores de escalas corrigidos no vetor de características de saída...
  for (i = 0, x = (escala_dominante * vect->nrotations); x < (vect->nscales * vect->nrotations); i++, x++) {
    vect_aux->mu[i] =  vect->mu[x];
    vect_aux->sigma[i] =  vect->sigma[x];
	}


  for (x = 0, i = (vect->nrotations * (vect->nscales - escala_dominante));  i < (vect->nscales * vect->nrotations); i++, x++) {
	  vect_aux->mu[i] =  vect->mu[x];
    vect_aux->sigma[i] =  vect->sigma[x];
	}

  for (x = 0; x < (vect->nscales * vect->nrotations); x++) {
    vect->mu[x] =  vect_aux->mu[x];
		vect->sigma[x] =  vect_aux->sigma[x];
  }

  DestroyVc(&vect_aux);
}

/*  Alinhamento do vetor de características.
 *  O alinhamento requer encontrar a escala e orientação dominante. */
void AlinhaDescritor(Vc * vect) {
  _AlinhaOrientacaoDescritor(vect);
  _AlinhaEscalaDescritor(vect);
}





double _fact(double n)
{
  double aux;
  int i;
  aux = 1.0;
	for (i = 2; i<=n; i++)
    aux *= (double) i;
	return aux;
}

double _FindAngle(int x1, int y1, int x2, int y2)
{
  /* pontos na mesma linha vertical */
  if ( x1 == x2 ) {
    if ( y1 == y2 ) return 0.0;         /* na verdade é indefinido, mas vamos
                                         * retornar 0.0 por padrao */
    else if ( y1 > y2 ) return M_PI_2;  /* ponto 1 está acima do ponto 2 */
    else return 3*M_PI_2;               /* ponto 1 está abaixo do ponto 2 */
  }
  /* pontos na mesma linha horizontal */
  if ( y1 == y2 ) {
    if ( x1 > x2 ) return 0.0;          /* ponto 1 está a direita do ponto 2 */
    if ( x1 < x2 ) return 2*M_PI;       /* ponto 1 está a esquerda do ponto 2 */
  }

  /* encontra o angulo (entre -pi e pi) */
  return atan2( (double)(y2-y1), (double)(x2-x1) ) + M_PI;
}


/******************************************************************************
 * Funções do filtro direcional que é uma reta
 ******************************************************************************/

bool _SValidPixel(Spectrum *flt, int x, int y)
{
  if ((x >= 0)&&(x < flt->ncols)&&
      (y >= 0)&&(y < flt->nrows))
    return(true);
  else
    return(false);
}


void _DrawSpectrumLineDDA(Spectrum *flt, Spectrum *spec, int x1, int y1, int xn, int yn){  
  int vx, vy;
  float Dx, Dy;
  int amostras; /* numero de pontos a serem pintados */
  float m; /* coeficiente angular da reta */
  int i;   
  float xk, yk;
  int p;


  vx = xn - x1; 
  vy = yn - y1;
  
   if(vx == 0){  
     Dx = 0.0;
     Dy = (float) SIGN(vy);
     amostras = abs(vy)+1;
   }
   else{  
     m = ((float)vy )/((float)vx);
     if( abs(vx) > abs(vy)){
       Dx = (float) SIGN(vx);
       Dy = m * Dx;
       amostras = abs(vx)+1; 
     }
     else{  
       Dy = (float) SIGN(vy);
       Dx = Dy / m;
       amostras = abs(vy)+1;
     }
   }
   

  xk = (float) x1;
  yk = (float) y1;
  for(i = 0; i < amostras; i++){ 
    if( _SValidPixel(flt, ROUND(xk), ROUND(yk)) ){
      p = ROUND(xk)+flt->tbrow[ROUND(yk)];
      flt->imag[p] = spec->imag[ROUND(xk)*flt->ncols+ROUND(yk)];
    }
    xk += Dx;
    yk += Dy;
  }
}


Spectrum *_FltDirecionalRETA(Image *img, double theta, int nrotations, int rotation)
{

  int x, y, x1, y1, x2, y2, nrows, ncols, p, pb;
  float r, fc;
  Spectrum *g, *spec;

  /* Enlarge ncols and nrows to the nearest power of 2 */
  ncols = (int)ceil(log(img->ncols)/log(2));
  nrows = (int)ceil(log(img->nrows)/log(2));
  ncols = 1 << ncols;
  nrows = 1 << nrows;
 
  g = CreateSpectrum(ncols, nrows);
  spec = CreateSpectrum(ncols, nrows);
  fc = MIN(img->ncols, img->ncols);

  /* usando a parte imaginaria para criar o circulo! centralizado */
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {

      p = x*ncols+y;
      r = (int) sqrt( (x-nrows/2)*(x-nrows/2) + (y-ncols/2)*(y-ncols/2) );

#if 0
      /* passa baixas v2*/
      if (r <= fc/2.0) {
        spec->imag[p] = 1.0;
      } else if (r < 3*fc/4.0) {
        spec->imag[p] = cos(M_PI_2 * log10(2.0*r/fc)/log10(2.0) );
      } else { /* r >= 3*fc/4.0 */
        spec->imag[p] = 0.0;
      }
#endif
#if LIGA_PASSA_BAIXA_DIRECIONAL
      /* passa baixas v1 */
      if (r <= fc/4.0) {
        spec->imag[p] = 1.0;
      } else if (r < fc/2.0) {
        spec->imag[p] = cos(M_PI_2 * log10(4.0*r/fc)/log10(2.0) );
      } else { /* r >= fc/2.0 */
        spec->imag[p] = 0.0;
      }
#else
        spec->imag[p] = 1.0;
#endif

      /* passa altas */
      if (r <= fc/4.0) {
        spec->imag[p] = 0.0;
      } else if ((r < fc/2.0) && (r > fc/4.0)) {
        spec->imag[p] *= cos(M_PI_2 * log10(2.0*r/fc)/log10(2.0) );
      }
    }
  }


  /* zerando a parte imaginária */
  memset( g->imag, '\0', sizeof(double)*ncols*nrows );

  /* criando o filtro e colocando na parte imaginária temporariamente */


  x1 = ncols/2;
  y1 = nrows/2;

  /* desenha a primeira parte! */
  if (fabs(cos(theta)) <= 10e-10) {
    x2 = x1;
    if (sin(theta)<0)
      y2 = nrows-1;
    else
      y2 = 0;
  }
  else {
    if (cos(theta) > 0)
      x2 = ncols -1;
    else
      x2 = 0;
    y2 = y1 - (x2-x1)*tan(theta);
  }

  _DrawSpectrumLineDDA(g, spec, x1, y1, x2, y2);

  x1 = ncols/2;
  y1 = nrows/2;

  /* desenha a segunda parte */
  if (fabs(cos(theta)) <= 10e-10) {
    x2 = x1;
    if (sin(theta)>0)
      y2 = nrows-1;
    else
      y2 = 0;
  }
  else {
    if (cos(theta) < 0)
      x2 = ncols -1;
    else
      x2 = 0;
    y2 = y1 - (x2-x1)*tan(theta);
  }
  _DrawSpectrumLineDDA(g, spec, x1, y1, x2, y2);

  /* libera memória do spectro de base */
  DestroySpectrum(&spec);


#if DESLOCA
  /* deslocando o filtro para o lugar certo, na parte real*/
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      pb = ((x+nrows/2)%nrows)*ncols + (y+ncols/2)%ncols;
      g->real[p] = g->imag[pb];
    }
  }
#else
  /* deslocando o filtro para o lugar certo, na parte real*/
  for (x=0; x<nrows; x++) {
    for (y=0; y<ncols; y++) {
      p = x*ncols+y;
      pb = x*ncols+y;
      g->real[p] = g->imag[pb];
    }
  }
#endif
  /* zerando a parte imaginária */
  memset( g->imag, '\0', sizeof(double)*ncols*nrows );


  _NormalizeSpectrum(g);

  return g;
}

/******************************************************************************
 * Funções Internas de Vetor Caracteristico direto do espaco de frequencias
 ******************************************************************************/
Spectrum *_SFiltraEscalaRot(Image *img, int nrotations, int rotation)
{
  Spectrum *flt;
  Spectrum *sresult;

  /* cria filtro direcional */
#if ESCOLHE_DIRECIONAL
  flt = _FltDirecional(img, ((float)rotation)*M_PI/((float) nrotations), nrotations, rotation);
#else
  flt = _FltDirecionalRETA(img, ((float)rotation)*M_PI/((float) nrotations), nrotations, rotation);
#endif


#if SAIDA_RELATORIO
  {
  Image *ig;
  ig = ViewMagnitude(flt);
  if (rotation==0) WriteImage(ig, "dirfilter0.pgm");
  if (rotation==1) WriteImage(ig, "dirfilter1.pgm");
  if (rotation==2) WriteImage(ig, "dirfilter2.pgm");
  if (rotation==3) WriteImage(ig, "dirfilter3.pgm");
  if (rotation==4) WriteImage(ig, "dirfilter4.pgm");
  if (rotation==5) WriteImage(ig, "dirfilter5.pgm");
  if (rotation==6) WriteImage(ig, "dirfilter6.pgm");
  DestroyImage(&ig);
  }
#endif


  /* guarda imagem para retornar */
  sresult = _SFilterImage(img, flt);

  /* apaga memória auxiliar */
  DestroySpectrum(&flt);

  return sresult;
}


Spectrum *_SFilterImage(Image *img, Spectrum *flt)
{
  Spectrum *simg, *sresult;

  /* imagem para o dominio da frequencia */
  simg = FFT2D(img);

  /* zerando a parte imaginária da imagem */
  //memset( fimg->imag, '\0', sizeof(double)*fimg->ncols*fimg->nrows );

  /* imagem filtrada com flt, no dominio da frequencia */
  sresult = MultSpectrum(simg,flt);

  /* apaga auxiliares */
  DestroySpectrum(&simg);

  /* retorna imagem filtrada no dominio do espaco */
  return sresult;
}


/*	Retorna a energia média acumulada da imagem. */
double _SGetMu(Spectrum *spec) {
  int i, n;
  double energia_acumulada = 0.0;
  n = spec->ncols * spec->nrows;
  
// calcula a energia acumulada... 
  for (i = 0; i < n; i++) {
    if (spec->real[i] != 0.0)
#if MU_QUAD
      energia_acumulada += pow(spec->real[i],2);
#else
      energia_acumulada += fabs(spec->real[i]);
#endif
  }
  return (energia_acumulada/n);
}

/* Retorna o desvio padrão da energia da imagem. */
double _SGetSigma(Spectrum *spec) {
  int i, n;
  double media = _SGetMu(spec), desvio = 0.0;
  
  n= spec->ncols*spec->nrows;
  for (i = 0; i < n; i++) {
    if (spec->real[i] != 0.0)
      desvio =+ pow(fabs(spec->real[i]) - media, 2);
  }
  return (sqrt(desvio/n));
}




/******************************************************************************
 * Funções do DB de imagens
 ******************************************************************************/


char *get_extension(char *fname)
{
  int i;
  char *p;
  p = fname;
  for (i=0;i<strlen(fname);i++)
    if (fname[i] == '.')
      p = fname + i*sizeof(char);
  return p;
}



/*  Retorna a soma dos valores absolutos das diferenças dos Vc`s. */
double CalcDistLinear(Vc *vect1, Vc *vect2) {
  Vc *vect_dif = CreateVc(vect1->nscales, vect1->nrotations);
  double soma = 0.0;
  int i; 
 
  for (i = 0; i < (vect1->nscales * vect1->nrotations); i++) {
    vect_dif->mu[i] = vect1->mu[i] - vect2->mu[i];
    vect_dif->sigma[i] = vect1->sigma[i] - vect2->sigma[i];
  }
  _Transf2Positive(vect_dif);

  for (i = 0; i < (vect1->nscales * vect1->nrotations); i++) {
    soma =+ vect_dif->mu[i] + vect_dif->sigma[i];
  }
  DestroyVc(&vect_dif);
  return (soma);
}

/*  Calcula a soma dos quadrados das diferencas (SSD) entre os dois vetores característicos. */
double CalcSumSquareDiff(Vc *vect1, Vc *vect2) {
  double soma = 0.0;
  int i; 

  for (i = 0; i < (vect1->nscales * vect1->nrotations); i++) {
    soma = pow(vect1->mu[i] - vect2->mu[i], 2) + pow(vect1->sigma[i] - vect2->sigma[i], 2);    
  }
  return soma;
}

/*  Retorna a distância euclidiana Vc`s. */
double CalcDistEuclid(Vc *vect1, Vc *vect2) {
  double soma = CalcSumSquareDiff(vect1, vect2);
  return sqrt(soma);
}


/*  Retorna um descritor positivo. */
void _Transf2Positive(Vc *vect) {
  double val_min_neg = _GetValMinNegative(vect);
  if (val_min_neg < 0.0) {
    double module = (-1) * val_min_neg;
    int i;
    for (i = 0; i < (vect->nscales * vect->nrotations); i++) {
      vect->mu[i] =+ module;
      vect->sigma[i] =+ module;
    }
  }
}

/*  Retorna o menor valor negativo entre os mu's e sigmas do descritor
 *  ou zero se todos forem positivos. */
double _GetValMinNegative(Vc *vect) {
    double vmn = 0.0;
    int i;
    for (i = 0; i < (vect->nscales * vect->nrotations); i++) {
      if (vect->mu[i] < vmn) {
        vmn = vect->mu[i];
      }
      if (vect->sigma[i] < vmn) {
        vmn = vect->sigma[i];      
      }
    }  
    return vmn;
}

/* EOF tira.c */

