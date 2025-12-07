#include "geometric3.h"
#include "algebra.h"
#include "interpolation3.h"
#include <math.h>


//Rotacao e translacao em volumes (3D)

void inversa(float M[4][4], float IM[4][4]){

	float detM;
	float aux[4][4],M_cofatores[4][4],Trans_cofat[4][4];
	int i,j;
	
	for(i=0;i<4;i++){
		for(j=0;j<4;j++){
			aux[i][j]=M[i][j];
			//printf("[%f] ",M[i][j]);
		}
		//printf("\n");
	}
			//calcula o determinante de M (a ultima linha de M é 0 0 0 1 - por laplace faco 1*Cofator M3x3)
	detM = aux[0][0]*aux[1][1]*aux[2][2] + aux[0][1]*aux[1][2]*aux[2][0] + aux[0][2]*aux[1][0]*aux[2][1]
			  -aux[0][0]*aux[1][2]*aux[2][1] - aux[0][1]*aux[1][0]*aux[2][2] - aux[0][2]*aux[1][1]*aux[2][0];
	
	
	if(detM==0)
		Error(MSG1,"Matriz Inversa (determinante==0)");
 
	// calculo da matriz dos cofatores
	for(i=0;i<4;i++){
		for(j=0;j<4;j++){
			M_cofatores[i][j]=Cofator(aux,i,j);
		}
	}

	//Transposta da Matriz dos cofatores
	TransMatrix(M_cofatores,Trans_cofat);

	//dividir a Matriz Transposta pelo determinante de M
	for(i=0;i<4;i++){
		for(j=0;j<4;j++){
			IM[i][j]=Trans_cofat[i][j]/detM;
		}

	}

}


float Cofator(float M[4][4], int l, int c){

	float aux[3][3];
	int linhas[3]={0,0,0},colunas[3]={0,0,0};
	int i,j;
	float cof=0;
	int aux_sinal;

	if(l==0){
		linhas[0]=1; linhas[1]=2; linhas[2]=3;
	} 
	else if(l==1){
		linhas[0]=0; linhas[1]=2; linhas[2]=3;
	}
	else if(l==2){
		linhas[0]=0; linhas[1]=1;linhas[2]=3;
	}
	else if(l==3){
		linhas[0]=0; linhas[1]=1;linhas[2]=2;
	}
	if(c==0){
		colunas[0]=1; colunas[1]=2; colunas[2]=3;
	} 
	else if(c==1){
		colunas[0]=0; colunas[1]=2;colunas[2]=3;
	}
	else if(c==2){
		colunas[0]=0; colunas[1]=1;colunas[2]=3;
	}
	else if(c==3){
		colunas[0]=0; colunas[1]=1;colunas[2]=2;
	}
	
	for(i=0;i<3;i++){
		for(j=0;j<3;j++){
			aux[i][j]=M[linhas[i]][colunas[j]];
		}
	}

	cof=aux[0][0]*aux[1][1]*aux[2][2] + aux[0][1]*aux[1][2]*aux[2][0] + aux[0][2]*aux[1][0]*aux[2][1]
		 -aux[0][0]*aux[1][2]*aux[2][1] - aux[0][1]*aux[1][0]*aux[2][2] - aux[0][2]*aux[1][1]*aux[2][0];
	
	aux_sinal=l+c;
	if ((aux_sinal%2) ==1)
		cof=-1*cof;
	return(cof);
}
void translacao(float T[4][4], float dx, float dy, float dz){
	
	T[0][0]=1;
	T[1][0]=0;
	T[2][0]=0;
	T[3][0]=0;
	
	T[0][1]=0;
	T[1][1]=1;
	T[2][1]=0;
	T[3][1]=0;
	
	T[0][2]=0;
	T[1][2]=0;
	T[2][2]=1;
	T[3][2]=0;
	
	T[0][3]=dx;
	T[1][3]=dy;
	T[2][3]=dz;
	T[3][3]=1;
}

void RotX(float Rx[4][4], float thx){

	float rad;
	if(thx<0)
		thx=360+thx;
	
	rad=thx*PI/180;
	Rx[0][0]=1;
	Rx[1][0]=0;
	Rx[2][0]=0;
	Rx[3][0]=0;
	
	Rx[0][1]=0;
	Rx[1][1]=cos(rad);
	Rx[2][1]=sin(rad);
	Rx[3][1]=0;
	
	Rx[0][2]=0;
	Rx[1][2]=-sin(rad);
	Rx[2][2]=cos(rad);
	Rx[3][2]=0;
	
	Rx[0][3]=0;
	Rx[1][3]=0;
	Rx[2][3]=0;
	Rx[3][3]=1;
}

void RotY(float Ry[4][4], float thy){

	float rad;
	if(thy<0)
		thy=360+thy;
	
	rad=thy*PI/180;
	Ry[0][0]=cos(rad);
	Ry[1][0]=0;
	Ry[2][0]=-sin(rad);
	Ry[3][0]=0;
	
	Ry[0][1]=0;
	Ry[1][1]=1;
	Ry[2][1]=0;
	Ry[3][1]=0;
	
	Ry[0][2]=sin(rad);
	Ry[1][2]=0;
	Ry[2][2]=cos(rad);
	Ry[3][2]=0;
	
	Ry[0][3]=0;
	Ry[1][3]=0;
	Ry[2][3]=0;
	Ry[3][3]=1;
}

void RotZ(float Rz[4][4], float thz){

	float rad;
	if(thz<0)
		thz=360+thz;
	
	rad=thz*PI/180;
	Rz[0][0]=cos(rad);
	Rz[1][0]=sin(rad);
	Rz[2][0]=0;
	Rz[3][0]=0;
	
	Rz[0][1]=-sin(rad);
	Rz[1][1]=cos(rad);
	Rz[2][1]=0;
	Rz[3][1]=0;
	
	Rz[0][2]=0;
	Rz[1][2]=0;
	Rz[2][2]=1;
	Rz[3][2]=0;
	
	Rz[0][3]=0;
	Rz[1][3]=0;
	Rz[2][3]=0;
	Rz[3][3]=1;
}

void transformScene(Scene *scn, float T[4][4], Scene *scn_out){
	int i, j, k;
      //  Scene *scn_out=CreateScene(scn->xsize,scn->ysize,scn->zsize);
        Point orig, dest;
        float IM[4][4];
        inversa(T, IM);
        for(k=0;k<scn_out->zsize;k++){
            for(j=0;j<scn_out->ysize;j++){
                for(i=0;i<scn_out->xsize;i++){
                    orig.x=i;
                    orig.y=j;
                    orig.z=k;
                    dest=TransformPoint(IM, orig);
                    if((dest.x>=0) && (dest.y>=0) && (dest.z>=0) && (dest.x<=scn->xsize-1) && (dest.y<=scn->ysize-1) && (dest.z<=scn->zsize-1)
                            && ValidVoxel(scn, (int)dest.x, (int)dest.y, (int)dest.z)){
                                       scn_out->data[scn_out->tbz[(int)orig.z]+scn_out->tby[(int)orig.y]+(int)orig.x]=(int)valueInterpol3(dest, scn);
                    }
                }
            }
        }
        scn_out->maxval=MaximumValue3(scn_out);
	scn_out->dx=scn->dx;
	scn_out->dy=scn->dy;
	scn_out->dz=scn->dz;
}

void transformacaoDireta(Scene *scn, float M[4][4], Scene *scn_out){
	int i,j,k;
	Point orig,dest;
	for(k=0;k<scn->zsize;k++){
		for(i=0;i<scn->xsize;i++){
			for(j=0;j<scn->ysize;j++){
			orig.x=i;
			orig.y=j;
			orig.z=k;
			dest=TransformPoint(M,orig);
			if(dest.x>=0&& dest.y>=0 && dest.z>=0 && ValidVoxel(scn_out,(int)dest.x,(int)dest.y,(int)dest.z))
				scn_out->data[scn_out->tbz[(int)dest.z]+scn_out->tby[(int)dest.y]+(int)dest.x]=
					scn->data[scn->tbz[(int)orig.z]+scn->tby[(int)orig.y]+(int)orig.x];
			}
		}
	}
	scn_out->maxval=MaximumValue3(scn_out);
}

void transformacaoMasc(Scene *scn, Scene *masc, float M[4][4], Scene *scn_out){ 
//transformacao com mascara, transforma apenas onde mascara==1
	int i,j,k;
	Point orig,dest;
	for(k=0;k<scn->zsize;k++){
		for(i=0;i<scn->xsize;i++){
			for(j=0;j<scn->ysize;j++){
				if(masc->data[masc->tbz[k]+masc->tby[j]+i] == 1){
					orig.x=i;
					orig.y=j;
					orig.z=k;
					dest=TransformPoint(M,orig);
					if(dest.x>=0&& dest.y>=0 && dest.z>=0 && ValidVoxel(scn_out,(int)dest.x,(int)dest.y,(int)dest.z))
						scn_out->data[scn_out->tbz[(int)dest.z]+scn_out->tby[(int)dest.y]+(int)dest.x]=
								scn->data[scn->tbz[(int)orig.z]+scn->tby[(int)orig.y]+(int)orig.x];
				
				}
			}
		}
	}
	scn_out->maxval=MaximumValue3(scn_out);
}
void limiar(Scene *scn, int thr){

	int i,n;
	n=scn->xsize*scn->ysize*scn->zsize;
	for(i=0;i<n;i++){
		if(scn->data[i]>=thr)
			scn->data[i]=1;
		else
			scn->data[i]=0;
	}
}
void transformScene_bin(Scene *scn,float T[4][4],Scene *out){

	Scene *bin255=CreateScene(scn->xsize,scn->ysize,scn->zsize);
	int n=scn->xsize*scn->ysize*scn->zsize;
	int i;
	for(i=0;i<n;i++){
		if(scn->data[i]==1)
			bin255->data[i]=255;
		else
			bin255->data[i]=0;
	}

	transformScene(bin255,T,out);
	limiar(out,128);
	out->maxval=MaximumValue3(out);
	DestroyScene(&bin255);
	
	
}
void Rotacao(float th_x,float th_y,float th_z, float T[4][4],Scene *scn,Scene *out){
	float Tc[4][4],To[4][4],Rz[4][4],Ry[4][4],Rx[4][4];
	float M1[4][4],M2[4][4],M3[4][4];
	
	//transladar o centro do objeto para a origem
	translacao(Tc,-scn->xsize/2,-scn->ysize/2,-scn->zsize/2);
	
	//Rotacoes nos eixos X, Y e Z
	RotX(Rx,th_x);
	RotY(Ry,th_y);
	RotZ(Rz,th_z);
	
	//tranladar para a o centro da imagem de saida
  translacao(To,out->xsize/2,out->ysize/2,out->zsize/2);

	//multiplicar as matrizes
	MultMatrices(To,Rz,M1);
	MultMatrices(M1,Ry,M2);
	MultMatrices(M2,Rx,M3);
	MultMatrices(M3,Tc,T);
  
}
void RotTrans(float th_x,float th_y,float th_z,float dx,float dy,float dz, float T[4][4],Scene *scn,Scene *out){
	float Tc[4][4],To[4][4],Rz[4][4],Ry[4][4],Rx[4][4],Trans[4][4];
	float M1[4][4],M2[4][4],M3[4][4],M4[4][4];
	
	//transladar o centro do objeto para a origem
	translacao(Tc,-scn->xsize/2,-scn->ysize/2,-scn->zsize/2);

	//Rotacoes nos eixos X, Y e Z
	RotX(Rx,th_x);
	RotY(Ry,th_y);
	RotZ(Rz,th_z);
	
	//tranladar para a o centro da imagem de saida (~ -tc)
  translacao(To,out->xsize/2,out->ysize/2,out->zsize/2);
  
	//Translacao em dx dy e dx
	translacao(Trans,dx,dy,dz); 
	
	//T= Trans To Rz Ry Rx Tc 
	

	//multiplicar as matrizes
	MultMatrices(Trans,To,M1);
	MultMatrices(M1,Rz,M2);
	MultMatrices(M2,Ry,M3);
	MultMatrices(M3,Rx,M4);
  MultMatrices(M4,Tc,T);
}
