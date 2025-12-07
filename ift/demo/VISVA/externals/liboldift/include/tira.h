#ifndef _TIRA_H_
#define _TIRA_H_

#include "image.h"
#include "spectrum.h"

/* aplica a equalização na imagem inicial - Deixa independente de
   luminosidade???!?!?!?!!?!?!?!?!?! */
#define EQUALIZA 1 //s1 f1

/* imprime os vetores de caracteristicas do db de imagens */
#define PRINT_DB 0

/* 1=cria imagens dos filtros pros relatorios */
#define SAIDA_RELATORIO 0

/* 1=energia é soma dos quadrados dos valores dos pixels, 0=energia é soma dos valores dos pixels */
#define MU_QUAD 0 //s1  f0 -

/* 1=usa uma DIMAGE para receber o resultado da multiplicação de espectros, 0=transforma direto pra Image */
#define USE_DIMAGE 0 //s0 f0

#define USE_IMAGE_ADD_MIN 0 //s0 f0

/* 1=retira energia e desvio no espaco, 0= retira energia e desvio na frequencia*/
#define FEATURES_NO_ESPACO 0 //s1 f0 - 



/* 1=passa baixas mais suave, 0=passa baixas mais abrupto */
#define ESCOLHE_PASSA_BAIXAS 0 //s0 f0



/* 1=angulo, 0=reta */
#define ESCOLHE_DIRECIONAL 1

/* 1=liga o passa baixa no direcional */
#define LIGA_PASSA_BAIXA_DIRECIONAL 1


typedef struct _Vc {
  double *mu, *sigma;
  int nscales;
  int nrotations;
} Vc;

typedef struct _InputDB {
  char *name;
  Image *img;
  Vc *vect;
  struct _InputDB *prox;
} InputDB;


/* rotinas para gerenciamento de Vetor de Características */
Vc  *CreateVc( int nscales, int nrotations );
void DestroyVc ( Vc **vect );
void PrintVc( Vc *vect );
void AlinhaDescritor(Vc * vect);

// Distancias

double CalcDistEuclid(Vc *vect1, Vc *vect2);
double CalcDistLinear(Vc *vect1, Vc *vect2);
double CalcSumSquareDiff(Vc *vect1, Vc *vect2);

/* extração do vetor de caracteristicas da textura de imagem */
Vc *Tira( Image *img, int nscales, int nrotations );

#endif /* _TIRA_H_ */
