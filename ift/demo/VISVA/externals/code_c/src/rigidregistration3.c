



#include <oldift.h>
#include "rigidregistration3.h"


timer *t1, *t2, *ti, *tf;
int n=7; //6 parametros
int m=6; //5 escalas



//// ------------------
// These functions are for registering left and right hemispheres
Scene *FlipSceneAxial(Scene *scn)
{
  Scene *res = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  Voxel u;
  int p,q;
  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	q = (scn->xsize-1-u.x) + scn->tby[u.y] + scn->tbz[u.z];
	res->data[q]=scn->data[p];
      }
  return res;
}

Scene *GetLeftAxialHalf(Scene *scn)
{
  Scene *res = CreateScene(scn->xsize/2,scn->ysize,scn->zsize);
  Voxel u;
  int p,q;
  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize/2; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	q = u.x + res->tby[u.y] + res->tbz[u.z];
	res->data[q]=scn->data[p];
      }
  return res;
}

Scene *GetRightAxialHalf(Scene *scn)
{
  Scene *res = CreateScene(scn->xsize/2,scn->ysize,scn->zsize);
  Voxel u;
  int p,q;
  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=(scn->xsize/2)+1; u.x < scn->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	q = u.x-scn->xsize/2-1 + res->tby[u.y] + res->tbz[u.z];
	res->data[q]=scn->data[p];
      }
  return res;
}

void MytransformScene(Scene *scn, float T[4][4], Scene *scn_out){
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
	  scn_out->data[scn_out->tbz[(int)orig.z]+scn_out->tby[(int)orig.y]+(int)orig.x]=(int)GetVoxelValue_trilinear(scn,dest.x,dest.y,dest.z);
	  //(int)valueInterpol3(dest, scn);
	}
      }
    }
  }
  scn_out->maxval=MaximumValue3(scn_out);
  scn_out->dx=scn->dx;
  scn_out->dy=scn->dy;
  scn_out->dz=scn->dz;
}

Scene *SelfRegisterAxial(Scene *scn, float T[4][4], float **best_theta)
{
  Scene *left,*right,*rightf,*newrightf,*newright;
  printf("Self Register: [pre-proc] "); fflush(stdout);
  left = GetLeftAxialHalf(scn);
  right = GetRightAxialHalf(scn);
  rightf = FlipSceneAxial(right);
  DestroyScene(&right);
  printf("[register] "); fflush(stdout);
  Register3(left,rightf,T,best_theta);  
  DestroyScene(&left);
  printf("[transform] "); fflush(stdout);
  newrightf=CreateScene(rightf->xsize,rightf->ysize,rightf->zsize);      
  transformScene(rightf,T,newrightf);
  DestroyScene(&rightf);
  newright=FlipSceneAxial(newrightf);
  DestroyScene(&newrightf);
  Scene *res = CopyScene(scn);
  Voxel u;
  int p,q;
  for (u.z=0; u.z < res->zsize; u.z++)
    for (u.y=0; u.y < res->ysize; u.y++)
       for (u.x=(res->xsize/2)+1; u.x < res->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	q = u.x-res->xsize/2-1 + newright->tby[u.y] + newright->tbz[u.z];
	res->data[p]=newright->data[q];
      }
  DestroyScene(&newright);
  printf("[done] \n"); fflush(stdout);
  return res;
}

//-----------------------








int MyOtsu(Scene *scn){
  // p: number of voxels; m: mean value; s: standard variation; J: score;
  Curve *hist=NormHistogram3(scn);
  double p1,p2,m1,m2,s1,s2,J,Jmax=-1.0;
  int i,T,Topt=0,Imax=MaximumValue3(scn);

  for (T=1; T < Imax; T++){
    p1 = 0.0;
    for (i=0; i <= T; i++) 
      p1 += hist->Y[i];
    p2 = 1.0 - p1;
    if ((p1 > 0.0)&&(p2 > 0.0)){
      m1 = 0.0;
      for (i=0; i <= T; i++) 
	m1 += hist->Y[i]*i;
      m1 /= p1;
      m2 = 0.0;
      for (i=T+1; i <= Imax; i++) 
	m2 += hist->Y[i]*i;
      m2 /= p2;
      s1 = 0.0;
      for (i=0; i <= T; i++) 
	s1 += hist->Y[i]*(i-m1)*(i-m1);
      s1 /= p1;
      s2 = 0.0;
      for (i=T+1; i <= Imax; i++) 
	s2 += hist->Y[i]*(i-m2)*(i-m2);
      s2 /= p2;
      J = (p1*p2*(m1-m2)*(m1-m2))/(p1*s1+p2*s2);
    }else{
      J = 0.0;      
    }
    if (J > Jmax){
      Jmax = J;
      Topt = T;
    }
  }
  DestroyCurve(&hist);
  return(Topt);
}
void MyMeansAboveBelowT(Scene *scn, int T, int *T1, int *T2){
  long long mean1=0,mean2=0,nv1=0,nv2=0;
  int p,n,delta;

  n  = scn->xsize*scn->ysize*scn->zsize;
  for(p=0; p < n; p++) {
    if (scn->data[p]<T){
      mean1+= scn->data[p];
      nv1++;
    }
    if (scn->data[p]>T){
      mean2+= scn->data[p];
      nv2++;
    }
  }

  delta = (int)(mean2/nv2 - mean1/nv1)/2;
  *T1   = T+delta;
  *T2   = T-delta;

  printf("T %d T1 %d T2 %d delta %d\n",T,*T1,*T2,delta);

}
Scene *MyApplySShape(Scene *scn, int a, int b, int c){
  Scene *enha=NULL;
  int p,n;
  float weight, Weight = 1.0;

  enha = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n    = scn->xsize*scn->ysize*scn->zsize;

  for (p=0; p < n; p++) {
    if (scn->data[p]<=a){
      weight = 0.0;
    }else{
      if ((scn->data[p]>a)&&(scn->data[p]<=b)){
	weight = (2.0*((float)scn->data[p]-a)*((float)scn->data[p]-a)/(((float)c-a)*((float)c-a)));
      }else{
	if ((scn->data[p]>b)&&(scn->data[p]<=c)){
	  weight = (1.0 - 2.0*((float)scn->data[p]-c)*((float)scn->data[p]-c)/(((float)c-a)*((float)c-a)));
	}else{
	  /* Felipe: weight = 2.0 remove o seio sagital (a veia do topo) das cenas de controle e weight = 1.0 nao consegue remover. Porem, weight = 1 gera accuracy bem maior nos phantoms do que weight = 2.  */
	  weight = Weight; 
	}
      }
      
    }
    enha->data[p]=(int)(scn->data[p]*weight);
  }
  return(enha);
}
Scene *enhance(Scene *in) {
  Scene *enha = NULL;
  int a,b,c;
  b = MyOtsu(in);
  MyMeansAboveBelowT(in,b,&c,&a);
  enha  = MyApplySShape(in,a,b,c);
  return enha;
}





Scene *NormalizeAccHist3(Scene *scn) {
    Scene *out = CreateScene(scn->xsize, scn->ysize, scn->zsize);
    Curve *acc = NormAccHistogram3(scn);
    int max, i, n, min;
    for (i = acc->n - 1; acc->Y[i] > 0.991; i--);
    max = i;
    for (i = 1; acc->Y[i] < 0.1; i++);
    min = i;
  

    n = scn->xsize * scn->ysize * scn->zsize;
    for (i = 0; i < n; i++) {
        if (scn->data[i] < min)
            out->data[i] = 0;

        else if (scn->data[i] <= max) {
            out->data[i] = ((scn->data[i] - min)*4095) / (max - min);
        } else
            out->data[i] = 4095;
    }
    DestroyCurve(&acc);
    return (out);
}
float Distance(float *f1, float *f2, int n){
  int i;
  float dist;

  dist = 0;
  for (i=0; i < n; i++)
    dist += (f2[i]-f1[i]);//(f2[i]-f1[i])/2.0;
  //dist /= n;

  return(dist);//exp(-(dist-0.5)*(dist-0.5)/2.0));

}
Scene *TextGradient(Scene *scn){
  float   dist,gx,gy,gz;
  int     i,p,q,n=scn->xsize*scn->ysize*scn->zsize;//,Imax=MaximumValue3(scn);
  Voxel   u,v;
  AdjRel3 *A=Spheric(1.0),*A6=Spheric(1.0);
  float   *mg=AllocFloatArray(A6->n);
  Scene   *grad=CreateScene(scn->xsize,scn->ysize,scn->zsize);


  typedef struct _features {
    float *f;
  } Features;

  Features *feat=(Features *)calloc(n,sizeof(Features));
  for (p=0; p < n; p++)
    feat[p].f = AllocFloatArray(A->n);

  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++) {
        p = u.x + scn->tby[u.y] + scn->tbz[u.z];
        for (i=0; i < A->n; i++) {
          v.x = u.x + A->dx[i];
          v.y = u.y + A->dy[i];
          v.z = u.z + A->dz[i];
          if (ValidVoxel(scn,v.x,v.y,v.z)){
            q = v.x + scn->tby[v.y] + scn->tbz[v.z];
            feat[p].f[i]=(float)scn->data[q];///(float)Imax;
          }
        }
      }

  for (i=0; i < A6->n; i++)
    mg[i]=sqrt(A6->dx[i]*A6->dx[i]+A6->dy[i]*A6->dy[i]+A6->dz[i]*A6->dz[i]);

  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++) {
        p = u.x + scn->tby[u.y] + scn->tbz[u.z];
        gx = gy = gz = 0.0;
        for (i=1; i < A6->n; i++) {
          v.x = u.x + A6->dx[i];
          v.y = u.y + A6->dy[i];
          v.z = u.z + A6->dz[i];
          if (ValidVoxel(scn,v.x,v.y,v.z)){
            q = v.x + scn->tby[v.y] + scn->tbz[v.z];
            dist = Distance(feat[p].f,feat[q].f,A->n);
            gx  += dist*A6->dx[i]/mg[i];
            gy  += dist*A6->dy[i]/mg[i];
            gz  += dist*A6->dz[i]/mg[i];
          }
        }
        grad->data[p]=(int)sqrt(gx*gx + gy*gy + gz*gz);//(100000.0*sqrt(gx*gx + gy*gy + gz*gz));
      }


  for (p=0; p < n; p++)
    free(feat[p].f);
  free(feat);

  free(mg);
  DestroyAdjRel3(&A);
  DestroyAdjRel3(&A6);
  return(grad);
}
Scene * getWaterGray3(Scene *scn, float di, float factor){
  Scene *grad=NULL,*handicap=NULL;
  Scene    *label=NULL;

  AdjRel3   *A=NULL;

  grad  = TextGradient(scn);

  A = Spheric(di); 
  handicap = Add3(grad,(int)(factor*MaximumValue3(grad)));
  label = WaterGray3(grad,handicap,A);
  
  DestroyAdjRel3(&A);
  DestroyScene(&grad);  
  DestroyScene(&handicap);  
  
  return(label);
}
Scene *getWGBorder(Scene *label){

  //Scene *label=getWaterGray3(scn,1.8,0.02);
	
  Scene *hscn=CreateScene(label->xsize,label->ysize,label->zsize);
  int p,q,i;
  AdjRel3 *A=NULL;
  Voxel u,v;

  A    = Spheric(1.0);
  for (u.z=0; u.z < hscn->zsize; u.z++){
    for (u.y=0; u.y < hscn->ysize; u.y++){
      for (u.x=0; u.x < hscn->xsize; u.x++){
	p = u.x + hscn->tby[u.y] + hscn->tbz[u.z];
	for (i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(hscn,v.x,v.y,v.z)){
	    q = v.x + hscn->tby[v.y] + hscn->tbz[v.z];
	    if (label->data[p] < label->data[q]){
	      hscn->data[p] = 1;
	      break;
	    }
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  return(hscn);
}	
Set *getSetBorder(Scene *scn){
    
    AdjRel3 *adj=Spheric(1.0);
    Scene *borda=GetBorder3(scn, adj);
    Set *S=NULL;
    int i, n;
    n=scn->xsize*scn->ysize*scn->zsize;
    for(i=0;i<n;i++){
        if(borda->data[i]==1){
            InsertSet(&S, i);
        }
    }
    DestroyScene(&borda);
    DestroyAdjRel3(&adj);
    return S;
}

Point3 CalcCenterOfGravity(Set *Sj, Scene *scn) {
    Set *seed = Sj;
    int p, z, y, x;
    Point3 soma, centro;
    soma.x = 0.;
    soma.y = 0.;
    soma.z = 0.;
    int n = 0;
    while (seed != NULL) {
        n++;
        p = seed->elem;
        z = (float) p / (scn->xsize * scn->ysize);
        y = (float) (p - scn->tbz[z]) / (scn->xsize);
        x = (p - scn->tbz[z]) % (scn->xsize);
        soma.z += z;
        soma.y += y;
        soma.x += x;
        seed = seed->next;
    }
    centro.x = (float) soma.x / n;
    centro.y = (float) soma.y / n;
    centro.z = (float) soma.z / n;
    DestroySet(&seed);
   
    return centro;
}

void calcTransformation(float *theta, float RT[4][4], float xc, float yc, float zc) {
    //calcula a matriz de tranformacao (rotacao e translacao) RT a partir dos parametros theta[i]
    // a transformacao é centrada no ponto de centro de gravidade (xc,yc,zc)
    
    //M= T -Tc R Tc
    float thx, thy, thz, X, Y, Z;
    thx = theta[0];
    thy = theta[1];
    thz = theta[2];
    if (thx < 0)
        thx = 360 + thx;
    if (thy < 0)
        thy = 360 + thy;
    if (thz < 0)
        thz = 360 + thz;
    //radianos
    X = thx * PI / 180;
    Y = thy * PI / 180;
    Z = thz * PI / 180;

    float T[4][4];
    T[0][0] = 1.0;
    T[0][1] = 0.0;
    T[0][2] = 0.0;
    T[0][3] = theta[3];
    T[1][0] = 0.0;
    T[1][1] = 1.0;
    T[1][2] = 0.0;
    T[1][3] = theta[4];
    T[2][0] = 0.0;
    T[2][1] = 0.0;
    T[2][2] = 1.0;
    T[2][3] = theta[5];
    T[3][0] = 0.0;
    T[3][1] = 0.0;
    T[3][2] = 0.0;
    T[3][3] = 1.0;
    float S=theta[6];
    
    float R[4][4];

    R[0][0] = S*(cos(Z) * cos(Y));
    R[1][0] = S*(sin(Z) * cos(Y) * cos(X) - sin(Y) * sin(X));
    R[2][0] = S*(-1. * sin(Z) * cos(Y) * sin(X) - sin(Y) * cos(X));
    R[3][0] = 0.0;

    R[0][1] = S*(-1. * sin(Z));
    R[1][1] = S*(cos(Z) * cos(X));
    R[2][1] = S*(-1. * cos(Z) * sin(X));
    R[3][1] = 0.0;

    R[0][2] = S*(cos(Z) * sin(Y));
    R[1][2] = S*(sin(Z) * sin(Y) * cos(X) + cos(Y) * sin(X));
    R[2][2] = S*(-1. * sin(Z) * sin(Y) * sin(X) + cos(Y) * cos(X));
    R[3][2] = 0.0;

    R[0][3] = 0.0;
    R[1][3] = 0.0;
    R[2][3] = 0.0;
    R[3][3] = 1.0;

    float TC[4][4];
    TC[0][0] = 1.0;
    TC[0][1] = 0.0;
    TC[0][2] = 0.0;
    TC[0][3] = (float) - 1 * xc;
    TC[1][0] = 0.0;
    TC[1][1] = 1.0;
    TC[1][2] = 0.0;
    TC[1][3] = (float) - 1 * yc;
    TC[2][0] = 0.0;
    TC[2][1] = 0.0;
    TC[2][2] = 1.0;
    TC[2][3] = (float) - 1 * zc;
    TC[3][0] = 0.0;
    TC[3][1] = 0.0;
    TC[3][2] = 0.0;
    TC[3][3] = 1.0;

    float T_C[4][4]; // -TC
    T_C[0][0] = 1.0;
    T_C[0][1] = 0.0;
    T_C[0][2] = 0.0;
    T_C[0][3] = (float) xc;
    T_C[1][0] = 0.0;
    T_C[1][1] = 1.0;
    T_C[1][2] = 0.0;
    T_C[1][3] = (float) yc;
    T_C[2][0] = 0.0;
    T_C[2][1] = 0.0;
    T_C[2][2] = 1.0;
    T_C[2][3] = (float) zc;
    T_C[3][0] = 0.0;
    T_C[3][1] = 0.0;
    T_C[3][2] = 0.0;
    T_C[3][3] = 1.0;

    float m1[4][4], m2[4][4];
 
    //M= T -Tc R Tc
    MultMatrices(T, T_C, m1);
    MultMatrices(m1, R, m2);
    MultMatrices(m2, TC, RT);

   
}

void printMatrix(float t[4][4]) {
    int lin, col;

    for (lin = 0; lin < 4; lin++) {
        for (col = 0; col < 4; col++)
            printf("[%f] ", t[lin][col]);
        printf("\n");
    }
    fflush(stdout);
}

double CriteryFunction_registration_gradient(Scene *B, Set *Sj, Point3 centro, Scene *movel, float *theta, float *delta, int i, int dir){
    float T[4][4];
    float *theta_aux=AllocFloatArray(n);
    double D=0.0;
    
    int k;
    if (delta==NULL){
       calcTransformation(theta, T, centro.x, centro.y, centro.z);
      // printMatrix(T);
    }
    else{
        if (i==-1){//aplica a descida de gradiente em todos os parametros
            for (k=0;k<n-1;k++)
                theta_aux[k]=theta[k]+delta[k];
            theta_aux[k]=theta[k]*delta[k]; //parametro de escala
        }
        else{
            for(k=0;k<n;k++)
                theta_aux[k]=theta[k];
            if(i!=(n-1)) //aplica a descida de graiente nos parametros de rotacao e escala
                theta_aux[i]=theta[i]+dir*delta[i];
            else{ //aplica a descida de graiente no parametro de escala
                if(dir==1)
                    theta_aux[i]=theta[i]*delta[i];
                else
                    theta_aux[i]=theta[i]/delta[i];
            }
        }
       calcTransformation(theta_aux, T, centro.x, centro.y, centro.z);
    }
    //calcula T(Sj) e a distancia D ao mesmo tempo
    Set *seed = Sj;
    int p;
    Voxel v, v_transf;
    while (seed != NULL) {
        p = seed->elem;
        v.z = p / (movel->xsize * movel->ysize);
        v.y = (p - movel->tbz[v.z]) / (movel->xsize);
        v.x = (p - movel->tbz[v.z]) % (movel->xsize);
        v_transf = Transform_Voxel(T, v);
        if (ValidVoxel(B, v_transf.x, v_transf.y, v_transf.z))
            D += B->data[B->tbz[v_transf.z] + B->tby[v_transf.y] + v_transf.x];
        seed = seed->next;
    }
 
    free(theta_aux);
    DestroySet(&seed);
    return D;
}

float * buscaMSGD(float **Delta, Scene *B, Set *Sj,Point3 centro,Scene *movel){
 
  ///printf("\nprocessando...\n");
    float *delta_best=AllocFloatArray(n);
    float *theta_best=AllocFloatArray(n);
    float VF_best, VF, VF0, VF1, VF2;
    int i,j;
    t1=Tic();
    
    float *theta=AllocFloatArray(n);
    for (i=0;i<n-1;i++)
      theta[i]=0.0;
    theta[i]=1.0;

    VF_best=CriteryFunction_registration_gradient(B,Sj,centro,movel,theta,NULL,-1,-1);
   
    for (i=0;i<n;i++)
        theta_best[i]=theta[i];
    do{
        VF0=VF_best;
        for(i=0;i<n;i++)
            theta[i]=theta_best[i];
        for(j=0;j<m;j++){
            
            for(i=0;i<n;i++){
                VF=VF0;
                
                if(i==n-1) delta_best[i]=1.0;
                else delta_best[i]=0;
                
                VF1=CriteryFunction_registration_gradient(B,Sj,centro,movel,theta,Delta[j],i,1); 
                VF2=CriteryFunction_registration_gradient(B,Sj,centro,movel,theta,Delta[j],i,-1); 
                if(VF1>VF){
                    VF=VF1;
                    delta_best[i]=Delta[j][i];
                }
                if(VF2>VF){
                    if(i==(n-1)){
                    delta_best[i]=1./Delta[j][i];
                    }
                    else
                    delta_best[i]=-1*Delta[j][i];
                }
                
           }
            VF=CriteryFunction_registration_gradient(B,Sj,centro,movel,theta,delta_best,-1,-1);
            if(VF>VF_best){
                VF_best=VF;
                for(i=0;i<n-1;i++)
                    theta_best[i]=theta[i]+delta_best[i];
                theta_best[i]=theta[i]*delta_best[i];
                //printf("RX/%f RY/%f RZ/%f TX/%f TY/%f TZ/%f S/%f: %f\n",delta_best[0],delta_best[1],delta_best[2],delta_best[3],delta_best[4],delta_best[5],delta_best[6],VF_best);
               
            }
        
        }

     
    }while(VF_best>VF0);
    t2=Toc();

    free(delta_best);
    free(theta);
    return(theta_best);
}


// input: fixa, movel
// output: T,best_theta
void Register3(Scene *fixa, Scene *movel, float T[4][4], float **best_theta)
{
  int i;
  //float T[4][4];  
  
  float **Delta = (float **) calloc(m,sizeof(float *));
  if(Delta==NULL) Error(MSG1,"AllocFloatMatrix");
  for(i=0;i<m;i++){
      Delta[i] =  (float *) calloc(n,sizeof(float));
      if (Delta[i]==NULL) Error(MSG1,"AllocFloatArray");
  }  
  
 
  Delta[0][0]=15.0;  //rx 
  Delta[0][1]=15.0;  //ry
  Delta[0][2]=15.0;  //rz
  Delta[0][3]=15.0;  //tx
  Delta[0][4]=15.0;  //ty
  Delta[0][5]=15.0;  //tz
  Delta[0][6]=1.0;   //s
 
  
  Delta[1][0]=10.0;
  Delta[1][1]=10.0;
  Delta[1][2]=10.0;
  Delta[1][3]=10.0;
  Delta[1][4]=10.0;
  Delta[1][5]=10.0;
  Delta[1][6]=1.0;   //s
  
  
  Delta[2][0]=5.00;
  Delta[2][1]=5.00;
  Delta[2][2]=5.00;
  Delta[2][3]=5.00;
  Delta[2][4]=5.00;
  Delta[2][5]=5.00;
  Delta[2][6]=1.0;   //s
  
  Delta[3][0]=1.0;
  Delta[3][1]=1.0;
  Delta[3][2]=1.0;
  Delta[3][3]=3.0;
  Delta[3][4]=3.0;
  Delta[3][5]=3.0;
  Delta[3][6]=1.0;   //s
 
  Delta[4][0]=0.5;
  Delta[4][1]=0.5;
  Delta[4][2]=0.5;
  Delta[4][3]=0.5;
  Delta[4][4]=0.5;
  Delta[4][5]=0.5;
  Delta[4][6]=1.0;   //s
 
  
  Delta[5][0]=0.1;
  Delta[5][1]=0.1;
  Delta[5][2]=0.1;
  Delta[5][3]=0.1;
  Delta[5][4]=0.1;
  Delta[5][5]=0.1;
  Delta[5][6]=1.0;   //s
  

  AdjRel3 *adj=NULL;
  Scene *B=NULL;
  Set *S=NULL;
  Point3 centro;
  Scene *fixa_norm=NULL, *movel_norm=NULL;
  Scene *movel_wg=NULL, *movel_wglines=NULL;
 
  //imagem de borda realcada 
  adj = Spheric(1.0);
  fixa_norm = NormalizeAccHist3(fixa);
  B  = TextGradient(fixa_norm);
  //B = MorphGrad3(fixa_norm,adj);
  DestroyAdjRel3(&adj);
  DestroyScene(&fixa_norm);
 
  //Conjunto de pontos (S) da imagem movel
  movel_norm = NormalizeAccHist3(movel); 
  movel_wg = getWaterGray3(movel_norm, 1.8, 0.07); 
  DestroyScene(&movel_norm);
  movel_wglines = getWGBorder(movel_wg);
  DestroyScene(&movel_wg);
  S = getSetBorder(movel_wglines);
  DestroyScene(&movel_wglines);
  
  centro = CalcCenterOfGravity(S, movel);
  //centro.x = movel->xsize/2;
  //centro.y = movel->ysize/2;
  //centro.z = movel->zsize/2;

  *best_theta=buscaMSGD(Delta, B,S,centro,movel);

  DestroyScene(&B);
  DestroySet(&S);
  for(i=0;i<m;i++)
    free(Delta[i]);
  free(Delta);

  
  //printf("Parametros finais: RX/%f RY/%f RZ/%f TX/%f TY/%f TZ/%f S/%f\n",best_theta[0],best_theta[1],best_theta[2],best_theta[3],best_theta[4],best_theta[5],best_theta[6]);
  
  calcTransformation(*best_theta,T,centro.x,centro.y,centro.z);

}

#ifdef _RIGID_REGISTRATION_STANDALONE_
int main( int argc, char **argv ) {

  float min;
  float T[4][4];
  float *best_theta;
  Scene *scn = NULL;
  Scene *mask = NULL;
  Scene *fixed = NULL;
  Scene *scn_int = NULL;
  Scene *fixed_int = NULL;
  Scene *reg = NULL;
  int i;

  if( argc < 4 ) {
    printf( "Usage must be: %s <fixed_scene> <input_scene> <output_scene> [<mask1_input> <mask1_output> <mask2_input> <mask2_input> ...]\n", argv[ 0 ] );
    return 0;
  }
  fixed = ReadScene( argv[ 1 ] );
  scn = ReadScene( argv[ 2 ] );

  min = MIN( MIN( MIN ( MIN ( MIN( scn->dx, scn->dy ), scn->dz ), fixed->dx ), fixed->dy ), fixed->dz );

  fixed_int = LinearInterp( fixed, min, min, min );
  scn_int = LinearInterp( scn, min, min, min );
  
  Register3( fixed, scn, T, &best_theta );
  
  reg = CreateScene( scn->xsize, scn->ysize, scn->zsize );
  transformScene( scn, T, reg );
  WriteScene( reg, argv[ 3 ] );
  DestroyScene( &reg );

  for( i = 4; i < argc; i += 2 ) {
    mask = ReadScene( argv[ i ] );
    reg = CreateScene( mask->xsize, mask->ysize, mask->zsize );
    transformScene( mask, T, reg );
    WriteScene( reg, argv[ i + 1 ] );
    DestroyScene( &reg );
    DestroyScene( &mask );
  }
    
  DestroyScene( &fixed_int );
  DestroyScene( &scn_int );
  free( best_theta );

  return( 0 );
}
#endif






