#include "geometric2.h"
#include "geometric3.h"
#include "algebra.h"
#include "interpolation3.h"
#include <math.h>


void transformacao2(Image *img, float IM[4][4], Image *img_transf){

    int i,j;
    Point orig,dest;
    for(j=1;j<img_transf->nrows-1;j++){
        for(i=1;i<img_transf->ncols-1;i++){
            orig.x=i;
            orig.y=j;
            orig.z=1; //imagem 2D
            dest=TransformPoint(IM,orig);
            if((dest.x>=0)&&(dest.y>=0)&&(dest.x<=img->ncols-1)&&(dest.y<=img->nrows-1)&&ValidPixel(img,(int)dest.x,(int)dest.y)){
                img_transf->val[img_transf->tbrow[(int)orig.y]+(int)orig.x]=(int)valueInterpol(dest,img);
            }
        }
    }
}

void transformacaoDireta2(Image *img, float M[4][4], Image *img_transf){

int i,j;
Point orig,dest;
    for(i=0;i<img->ncols;i++){
        for(j=0;j<img->nrows;j++){
            orig.x=i;
            orig.y=j;
            orig.z=1;  //imagem 2D
            dest=TransformPoint(M,orig);
            if(ValidPixel(img_transf,(int)dest.x,(int)dest.y))
                img_transf->val[img_transf->tbrow[(int)dest.y]+(int)dest.x]=img->val[img->tbrow[(int)orig.y]+(int)orig.x];

        }
    }
}

void Rotacao2(float th, float T[4][4],Image *img ,Image *out){
	float Tc[4][4],To[4][4],R[4][4];
	float M1[4][4],M2[4][4];

	//transladar o centro do objeto para a origem
	translacao(Tc,-(int)img->ncols/2,-(int)img->nrows/2,0);

	//Rotacao
	RotZ(R,th);

	//tranladar para a o centro da imagem de saida
  translacao(To,(int)out->ncols/2,(int)out->nrows/2,0);

	//multiplicar as matrizes
	MultMatrices(To,R,M1);
	MultMatrices(M1,Tc,M2);
	inversa(M2,T);
}

void Translacao2(float dx,float dy, float T[4][4],Image *img, Image *out){

	float Tc[4][4],To[4][4],T1[4][4];
	float M1[4][4],M2[4][4];

	//transladar o centro do objeto para a origem
	translacao(Tc,-(int)img->ncols/2,-(int)img->nrows/2,0);

	//Rotacoes nos eixos X, Y e Z
	translacao(T1,dx,dy,0);

	//tranladar para a o centro da imagem de saida
    translacao(To,(int)out->ncols/2,(int)out->nrows/2,0);

	//multiplicar as matrizes
	MultMatrices(To,T1,M1);
	MultMatrices(M1,Tc,M2);

	inversa(M2,T);
}

void RotTrans2(float th,float dx,float dy,float T[4][4],Image *img,Image *out){
	float Tc[4][4],To[4][4],R[4][4],Trans[4][4];
	float M1[4][4],M2[4][4],M3[4][4];

	//transladar o centro do objeto para a origem
	translacao(Tc,-(int)img->ncols/2,-(int)img->nrows/2,0);

	//Rotacao
	RotZ(R,th);

	//Translacao em dx dy
	translacao(Trans,dx,dy,0);

	//tranladar para o centro da imagem de saida
  translacao(To,(int)out->ncols/2,(int)out->nrows/2,0);

	//multiplicar as matrizes
	MultMatrices(To,Trans,M1);
	MultMatrices(M1,R,M2);
	MultMatrices(M2,Tc,M3);
  inversa(M3,T);
}

void SclMatrix(float S[4][4], float sx, float sy, float sz)
{
	S[0][0]=sx;
	S[1][0]=0;
	S[2][0]=0;
	S[3][0]=0;

	S[0][1]=0;
	S[1][1]=sy;
	S[2][1]=0;
	S[3][1]=0;

	S[0][2]=0;
	S[1][2]=0;
	S[2][2]=sz;
	S[3][2]=0;

	S[0][3]=0;
	S[1][3]=0;
	S[2][3]=0;
	S[3][3]=1;
}

void RotTrans2DiffOrigin(float th,float dx,float dy,float T[4][4],Image *img,Image *out, float xo, float yo){
	float Tc[4][4],To[4][4],R[4][4],Trans[4][4];
	float M1[4][4],M2[4][4],M3[4][4];

	//transladar o centro do objeto para a origem
	translacao(Tc,-(int)xo,-(int)yo,0);

	//Rotacao
	RotZ(R,th);

	//Translacao em dx dy
	translacao(Trans,dx,dy,0);

	//tranladar para o centro da imagem de saida
    translacao(To,(int)xo,(int)yo,0);

	//multiplicar as matrizes
	MultMatrices(To,Trans,M1);
	MultMatrices(M1,R,M2);
	MultMatrices(M2,Tc,M3);
    inversa(M3,T);

}

void SclRotTrans2DiffOrigins(float sx, float sy, float th,float dx,float dy,float T[4][4],
    Image *img,Image *out, float xo, float yo, float x1, float y1){
	float Tc[4][4],To[4][4],R[4][4],Trans[4][4], S[4][4];
	float M1[4][4],M2[4][4],M3[4][4], M4[4][4];

	//transladar o centro do objeto para a origem
	translacao(Tc,-(int)xo,-(int)yo,0);

    // Escala
    SclMatrix(S, sx, sy, 1);

	//Rotacao
	RotZ(R,th);

	//Translacao em dx dy
	translacao(Trans,dx,dy,0);

	//tranladar para o centro da imagem de saida
    translacao(To,(int)x1,(int)y1,0);

	//multiplicar as matrizes
	MultMatrices(To,Trans,M1);
	MultMatrices(M1,R,M2);
    MultMatrices(M2,S,M3);

	MultMatrices(M3,Tc,M4);
    inversa(M4,T);

}

