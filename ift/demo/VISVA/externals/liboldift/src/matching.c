#include "matching.h"
#include "scene.h"
#include "matrix.h"
#include "curve.h"
#include "radiometric3.h"
#include "math.h"

double QuadraticMeanError(Scene *scn1, Scene *scn2){
	
	int n,i;
	double e=0.0;
	n=scn1->xsize*scn1->ysize*scn1->zsize;
	for(i=0;i<n;i++)
		e=e+pow((scn1->data[i]- scn2->data[i]),2);
	
	return e;
}

double MutualInformation(Scene *scn1, Scene *scn2){
	double mi=0.0;
	int L1,L2,l1,l2;
	Curve *h1, *h2;
	Matrix *h12=NormJoinHistogram(scn1,scn2);
	L1=scn1->maxval;
	L2=scn2->maxval;
	h1=NormHistogram3(scn1);// WriteCurve(h1,"H1.txt");
	h2=NormHistogram3(scn2);// WriteCurve(h2,"H2.txt");
	for(l1=0;l1<=L1;l1++){
		for(l2=0;l2<=L2;l2++){
			if(h1->Y[l1]* h2->Y[l2]!=0)
			mi=mi+( h12->val[h12->tbrow[l2]+l1] * log (1+ h12->val[h12->tbrow[l2]+l1]/( h1->Y[l1]* h2->Y[l2]  )));
		}
	}
	DestroyMatrix(&h12);
	DestroyCurve(&h1);
	DestroyCurve(&h2);
	//printf("mi: %f\n",mi);
	return mi;	
}


