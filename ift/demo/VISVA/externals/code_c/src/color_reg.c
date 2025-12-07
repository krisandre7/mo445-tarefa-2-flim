
#include <oldift.h>
#include "color_reg.h"



Scene *MyNormalizeAccHist3(Scene *scn){
	Scene *out=CreateScene(scn->xsize,scn->ysize,scn->zsize);
	Curve *acc=NormAccHistogram3(scn);
	int max,i,n,min;
	for(i=acc->n -1; acc->Y[i] >0.991; i--);
	max=i;
	//printf("99 = %d\n",i);
	
	for(i=1; acc->Y[i] <0.1;i++);
	min=i;
	//printf("1 = %d\n",min);
	
	n=scn->xsize*scn->ysize*scn->zsize;
	for(i=0;i<n;i++){
		if(scn->data[i]<min)
			out->data[i]=0;
			
		else if(scn->data[i]<=max){
			out->data[i]=((scn->data[i]-min)*4095)/(max-min);
		}
		else
			out->data[i]=4095;
	}
	DestroyCurve(&acc);
	return(out); 
}





void ColorReg(Scene *scn1, Scene *scn2, Scene **R, Scene **G, Scene **B)
{

    Scene *scn1_norm=MyNormalizeAccHist3(scn1);
    Scene *scn2_norm=MyNormalizeAccHist3(scn2); 

    *R=CreateScene(scn1->xsize,scn1->ysize,scn1->zsize);
    *G=CreateScene(scn1->xsize,scn1->ysize,scn1->zsize);
    *B=CreateScene(scn1->xsize,scn1->ysize,scn1->zsize);


    Voxel u;
    int p,q;
    for (u.z=0; u.z < scn1_norm->zsize; u.z++)
	for (u.y=0; u.y < scn1_norm->ysize; u.y++)
	    for (u.x=0; u.x < scn1_norm->xsize; u.x++){
		p = u.x + scn1_norm->tby[u.y] + scn1_norm->tbz[u.z];
		(*R)->data[p]=scn1_norm->data[p];
		if (ValidVoxel(scn2_norm,u.x,u.y,u.z)) {
		  q = u.x + scn2_norm->tby[u.y] + scn2_norm->tbz[u.z];
		  (*G)->data[p]=scn2_norm->data[q];
		  (*B)->data[p]=(scn1_norm->data[p]+scn2_norm->data[q])/2;
		} else {
		  (*G)->data[p]=0;
		  (*B)->data[p]=scn1_norm->data[p]/2;
		}
	    }

}




