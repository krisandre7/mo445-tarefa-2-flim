#include "ventws.h"

/* int otsu=0; */

int AscSort(  void * x,   void * y) {
    Hist *xx = (  Hist *)x;
    Hist *yy = (  Hist *)y;
  return (xx->nitems - yy->nitems);
}

int NumericSort(  void * x,   void * y) {
    Hist *xx = (  Hist *)x;
    Hist *yy = (  Hist *)y;
  return (yy->nitems - xx->nitems);
}

int DescSort(  void * x,   void * y) {
    Hist *xx = (  Hist *)x;
    Hist *yy = (  Hist *)y;
  return (yy->nitems - xx->nitems);
}


/* pt2Func MySort(char opCode) { */
/*   /\* */
/*     opCode: */
/*      == 0 - descrescente */
/*      != 0 - crescente */
/*    *\/ */

/*   return (opCode) ? &AscSort : &DescSort; */

/* } */

int EucDist3D(int x, int y, int z, DVoxel v) {    
  return (int)sqrt(pow(x - v.x, 2) + pow(y - v.y, 2) + pow(z - v.z, 2));
}

void FreeHist(Hist **hist) {

  Hist *aux;

  aux = *hist;

  if (aux != NULL)
    free(aux);
  *hist = NULL;

}

int MyPlaneInfo(const char *filename, PlaneInfo *plane) {

  char fname[100] = "";
  FILE *fp;

  sprintf(fname, "%s.info", filename);
  if ((fp = fopen(fname, "rb")) == NULL) {
    printf("[MyPlaneInfo] Cannot open file %s for reading\n", fname);
    return 0;
  }

  fread(plane, sizeof(Plane), 1, fp);

  fclose(fp);

  return (1);

}
  
Elems Parse(char *arg) {

  Elems elems;
  char s='-', *str, *tmp;
  int pos = 0, len;

  len = strlen(arg);
  str = (char *)malloc(len*sizeof(char));
  tmp = (char *)malloc(len*sizeof(char));

  strcpy(str, arg);
  strcpy(tmp, arg);

  elems.first = -1;
  elems.last = -1;

  while ((*str++ != s) && (*str))
    pos++;

  /* Separador pode estar na �ltima posi��o */
  if (pos == (len-1)) {
    /* N�o existe separador */
    if (arg[pos] != s)
      elems.first = elems.last = atoi(arg);
    /* Separador est� na �ltima posi��o */
    else if (len > 1){
      strncpy(str, arg, pos);
      elems.first = atoi(str);
    }
  }
  /* Separador est� na primeira posicao */
  else if ((pos == 0) && (len > 1)) {
    elems.first = 0;
    elems.last = atoi(str);
  }
  /* Separador no meio da string */
  else {
    strncpy(tmp, arg, pos);
    elems.first = atoi(tmp);
    elems.last = atoi(str);
  }

  str = tmp = NULL;
  free(str);
  free(tmp);

  return(elems);

}


Hist * ClosestToCenter(Scene *bin, PlaneInfo *plane, AdjRel3 *A) {
  /*
    ATENCAO
    Nesta funcao, dist->nitems nao armazena o numero de voxels do componente e
    sim a distancia euclidiana em relacao ao centro de gravidade do objeto
   */  

  int i, j, k, d, max, p;
  Hist *dist=NULL, htmp;
  Scene *label=NULL;  
  
  label = LabelBinComp3(bin, A);
  max = MaximumValue3(label)+1;

  dist = (Hist *)malloc(max*sizeof(Hist));

  for (i = 0; i < max; i++) {
    dist[i].nitems = INT_MAX;
    dist[i].val = i;
  }

  for (i = 0; i < label->xsize; i++)
    for (j = 0; j < label->ysize; j++)
      for (k = 0; k < label->zsize; k++) {
        d = EucDist3D(i, j, k, plane->center);
        p = label->data[i + label->tby[j] + label->tbz[k]];
        if (d < dist[p].nitems)
          dist[p].nitems = d;
      }
  
  /* MySort: 1-crescente, 0-decrescente */
/*   qsort(dist, (size_t)max, sizeof(Hist), (void *)AscSort); */
  qsort(dist, (size_t)max, sizeof(Hist), AscSort);

  /* O fundo deve ficar em primeiro, ent�o troca-se com o primeiro elemento */
  if (dist[1].val == 0) {
    htmp = dist[0];
    dist[0].val = dist[1].val;
    dist[0].nitems = dist[1].nitems;
    dist[1].val = htmp.val;
    dist[1].nitems = htmp.nitems;
  }
  
  DestroyScene(&label);
  return (dist);

}

/*
  Retorna uma cena com os 'ncomps' maiores componentes
*/
Scene * LargestSceneComp(Scene *comp, int ncomps) {

  Scene *scn=NULL;
  int i, j, n, max=MaximumValue3(comp)+1;
  Hist *hist=NULL;

  hist = (Hist *)malloc(max*sizeof(Hist));

  scn = CreateScene(comp->xsize, comp->ysize, comp->zsize);

  n = comp->xsize*comp->ysize*comp->zsize;

  for (i = 0; i < max; i++) {
    hist[i].val = i;
    hist[i].nitems = 0;
  }

  /* Computa o n�mero de voxels de cada componente (menos do fundo) */
  for (i = 0; i < n; i++)
    if (comp->data[i] != 0)
      hist[comp->data[i]].nitems++;
  
/*   qsort(hist, (size_t)max, sizeof(Hist), (void *)NumericSort); */
  qsort(hist, (size_t)max, sizeof(Hist), NumericSort);
  
  for (j = 0; j < ncomps; j++) {
    for (i = 0; i < n; i++) {
      if (comp->data[i] == hist[j].val) {
        scn->data[i] = 1;
      }
    }
  }
  
  FreeHist(&hist);
  return(scn);
}


void RemoveCenterVoxels(Scene *bin, PlaneInfo *plane) {
  
  Scene *scn=NULL;
  int i, j, k, l, r, s, max, p, nvoxels;
  double left, right;
  DVoxel v;
  Image *slice=NULL, *icomp=NULL;
  Comp *comp = NULL;
  AdjRel *A=Circular(1.0);
  
  scn = CopyScene(bin);

  for (k = 0; k < bin->zsize; k++) {
    slice = GetSlice(bin, k);
    icomp = LabelBinComp(slice, A);
    max = MaximumValue(icomp)+1;
    if (max > 1) {
      comp = (Comp *)malloc(max*sizeof(Comp));
      for (l = 0; l < max; l++) {
        comp[l].left = 0;
        comp[l].right = 0;
      }
      for (i = 0; i < bin->xsize; i++) {
        for (j = 0; j < bin->ysize; j++) {
          v.x = (double)i - plane->center.x;
          v.y = (double)j - plane->center.y;
          v.z = (double)k - plane->center.z;
          p = i + icomp->tbrow[j];
          if (icomp->val[p] != 0) {
            if (InternalProduct(plane->normal, v) < 0.0) {
              comp[icomp->val[p]].left++;
            }
            else {
              comp[icomp->val[p]].right++;
            }
            comp[icomp->val[p]].val = icomp->val[p];
          }
        }
      }
      for (l = 0; l < max; l++) {
        left = (double)comp[l].left;
        right = (double)comp[l].right;
        if ((left > 0) && (right > 0) &&
            (((left/right > 0.2) && (left/right <= 1.0)) ||
             ((right/left > 0.2) && (right/left <= 1.0)) )) {
          for (r = 0; r < icomp->ncols; r++) {
            for (s = 0; s < icomp->nrows; s++) {
              if (icomp->val[r + icomp->tbrow[s]] == comp[l].val) {
                scn->data[r + scn->tby[s] + scn->tbz[k]] = 0;
              }
            }
          }
        }
      }
    }
    if (comp != NULL) {
      free(comp);
      comp = NULL;
    }
  }

  nvoxels = scn->xsize*scn->ysize*scn->zsize;
  for (i = 0; i < nvoxels; i++)
    bin->data[i] = scn->data[i];
  
  /* WriteScene(scn, "scn.scn"); */
  
  DestroyAdjRel(&A);
  DestroyImage(&slice);
  DestroyImage(&icomp);
  DestroyScene(&scn);
  
}


void RemoveBorderVoxels(Scene *bin, Scene *border) {
  
  /*   Scene *scn=NULL; */
  int i, j, k, l, p;
  AdjRel3 *A=Spheric(10.0);
  Voxel v;
  
  /* scn = CopyScene(bin); */
  for (i = 0; i < border->xsize; i++) {
    for (j = 0; j < border->ysize; j++) {
      for (k = 0; k < border->zsize; k++) {
        p = i + border->tby[j] + border->tbz[k];
        if (border->data[p] != 0) {
          for (l = 1; l < A->n; l++) {
            v.x = i + A->dx[l];
            v.y = j + A->dy[l];
            v.z = k + A->dz[l];
            /* u = A->dx[l] + A->dy[l] + A->dz[l]; */
            bin->data[v.x + bin->tby[v.y] + bin->tbz[v.z]] = 0;
          }
        }
      }
    }
  }
  
  DestroyAdjRel3(&A);
  
}


void MySelectLargestComp(Scene *ero)
{
  AdjRel3 *A=Spheric(1.8);
  Scene  *label=LabelBinComp3(ero,A);
  int Lmax=MaximumValue3(label);
  int *area=(int *)AllocIntArray(Lmax+1);
  int imax, i, p, n=ero->xsize*ero->ysize*ero->zsize;
  
  for (p=0; p < n; p++)
    if (label->data[p]>0)
      area[label->data[p]]++;
  imax = 0;
  for (i=1; i <= Lmax; i++)
    if (area[i]>area[imax])
      imax = i;
  for (p=0; p < n; p++)
    if (label->data[p]!=imax)
      ero->data[p]=0;
  DestroyScene(&label);
  DestroyAdjRel3(&A);
}


float MyDistance(float *f1, float *f2, int n)
{
  int i;
  float dist;
  
  dist = 0;
  for (i=0; i < n; i++)
    dist += (f2[i]-f1[i]);
  
  return(dist);
  
}


Scene *MyTextGradient(Scene *scn)
{
  float   dist,gx,gy,gz;
  int     i,p,q,n=scn->xsize*scn->ysize*scn->zsize;
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
            dist = MyDistance(feat[p].f,feat[q].f,A->n);
            gx  += dist*A6->dx[i]/mg[i];
            gy  += dist*A6->dy[i]/mg[i];
            gz  += dist*A6->dz[i]/mg[i];
          }
        }
        grad->data[p]=(int)sqrt(gx*gx + gy*gy + gz*gz);
      }
  
  
  for (p=0; p < n; p++)
    free(feat[p].f);
  free(feat);
  
  free(mg);
  DestroyAdjRel3(&A);
  DestroyAdjRel3(&A6);
  return(grad);
}


int MyOtsu(Scene *scn)
{
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

void MyMeansAboveBelowT(Scene *scn, int T, int *T1, int *T2)
{
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

  /* printf("T %d T1 %d T2 %d delta %d\n",T,*T1,*T2,delta); */

}

Scene *MyApplySShape(Scene *scn, int a, int b, int c)
{
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
	  weight = Weight;
	}
      }
      
    }
    enha->data[p]=(int)(scn->data[p]*weight);
  }
  return(enha);
}


Scene *enhance(Scene *in, int otsu) {
  Scene *enha = NULL;
  int a, c;
  MyMeansAboveBelowT(in, otsu, &c, &a);
  enha  = MyApplySShape(in, a, otsu, c);
  return enha;
}

Scene * GradientScene(Scene *in, int e, int otsu) {

  Scene *out=NULL, *enha=NULL;
  int i;

  if (e) enha=enhance(in, otsu); else { enha = in; in = NULL; }
  out = MyTextGradient(enha);
  DestroyScene(&enha);

  for(i = 0; i < out->xsize*out->ysize*out->zsize; i++) {
    if (out->data[i] < 0) out->data[i] = 0;
    if (out->data[i] > 32767) out->data[i] = 32767;
  }

  return (out);

}

Scene * BrainMarkerScene(Scene *scn, int otsu) {

  Scene *bin=NULL, *ero=NULL;
  Set  *S=NULL;

  bin = Threshold3(scn, otsu, 4095);
  ero = ErodeBin3(bin, &S, 5.0);
  DestroyScene(&bin);
  DestroySet(&S);
  MySelectLargestComp(ero);
  
  return (ero);
}

Scene * VentricleMarkerScene(Scene *scn, Elems elems, double adj,
                             int erode, PlaneInfo plane, int otsu) {

  Scene *ero = NULL, *label=NULL, *larg=NULL;
  Scene *ventricle=NULL;
  Scene *bin=NULL, *border=NULL, *comp=NULL;
  AdjRel3 *A=Spheric(adj);
  Set *S=NULL;
  int n, T;
  int i, j, tmp;
  float ofact;
  Hist *closest=NULL;

  ofact = 1.2;
/*   T = MyOtsu(scn); */
  T = otsu;
  
  ventricle = CreateScene(scn->xsize, scn->ysize, scn->zsize);

  n = ventricle->xsize*ventricle->ysize*ventricle->zsize;
  for (i = 0; i < n; i++) {
    if ((scn->data[i] < ofact*T) && (scn->data[i] != 0))
      ventricle->data[i] = 1;
  }

  bin = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  for (i = 0; i < n; i++)
    if (scn->data[i] != 0)
      bin->data[i] = 1;

  border = GetBorder3(bin, A);

  RemoveBorderVoxels(ventricle, border);
  RemoveCenterVoxels(ventricle, &plane);
  comp = LabelBinComp3(ventricle, A);
  
  larg = LargestSceneComp(comp, 5);

  for (i = 0; i < n; i++) {
    bin->data[i] = 0;
    if (larg->data[i] != 0)
      bin->data[i] = 1;
  }
  
  closest = ClosestToCenter(bin, &plane, A);
  comp = LabelBinComp3(bin, A);

  for (i = 0; i < n; i++)
    bin->data[i] = 0;

  tmp = MaximumValue3(comp);
  if (elems.last == -1)
    elems.last = tmp;

  if (closest) {
    for (i = 0; i < n; i++)
      for (j = elems.first; j <= elems.last; j++) {
        if (comp->data[i] == closest[j].val) {
          bin->data[i] = 1;
          break;
        }
      }
  }

  if (erode)
    ero = ErodeBin3(bin, &S, 1.0);
  else
    ero = CopyScene(bin);

  DestroyScene(&label);
  DestroyScene(&larg);
  DestroyScene(&bin);
  DestroyScene(&border);
  DestroyScene(&ventricle);
  DestroyScene(&comp);
  DestroySet(&S);
  FreeHist(&closest);
  DestroyAdjRel3(&A);

  return (ero);

}


Scene * StandardDeviation(Scene *scn, Scene *bin) {

  int i, j, k, l, p, q, n, nvoxels;
  double s, m;
  AdjRel3 *A = Spheric(1.0);
  Scene *border = NULL, *vent = NULL;
  Voxel v;

  n = scn->xsize*scn->ysize*scn->zsize;

  m = 0.0;
  nvoxels = 0;
  for (i = 0; i < n; i++)
    if (bin->data[i] != 0) {
      m += scn->data[i];
      nvoxels++;
    }

  /* M�dia do brilho dos voxels */
  m /= nvoxels;
  
  /* Vari�ncia */
  s = 0.0;
  for (i = 0; i < n; i++)
    if (bin->data[i] != 0) {
      s += scn->data[i]*scn->data[i];
    }

  s = sqrt((s - nvoxels*m*m)/(nvoxels-1));

  border = GetBorder3(bin, A);
  vent = CopyScene(bin);

  nvoxels = 0;
  for (i = 0; i < border->xsize; i++)
    for (j = 0; j < border->ysize; j++)
      for (k = 0; k < border->zsize; k++) {
        p = i + border->tby[j] + border->tbz[k];
        if (border->data[p] != 0) {
          for (l = 1; l < A->n; l++) {
            v.x = i + A->dx[l];
            v.y = j + A->dy[l];
            v.z = k + A->dz[l];
            q = v.x + border->tby[v.y] + border->tbz[v.z];
            if ((scn->data[q] < (m+s)) && (scn->data[q] > (m-s))
                && (vent->data[q] == 0)) {
              vent->data[q] = 1;
              nvoxels++;
            }
          }
          
        }
      }

  printf("NVOXELS: %d\n", nvoxels);

  DestroyScene(&border);
  DestroyAdjRel3(&A);

  return (vent);
}


Scene * GetVentricle (const char *basename,
                      Scene *iscn, double adj,
                      int erode, int debug) {

  Scene *grad, *bseed, *vseed;
  Scene *cost, *label, *pred;
  Set *S = NULL;
  AdjRel3 *A;
  int W, H, D, N, i, p, q, c1;
  Voxel v, u;
  Queue *Q;
  Elems elems;
  char filename[100];
  PlaneInfo plane;
  int otsu;

  printf("Adjacency Relation: %f\n", adj);
  printf("Erode Seed Set: %d\n", erode);

  /* Parse dos componentes a serem selecionados */
  elems.first = 1;
  elems.last = 2;

  otsu = MyOtsu(iscn);
  grad = GradientScene(iscn, 1, otsu); /* 1:enhance, 0:don't enhance */
  bseed = BrainMarkerScene(iscn, otsu);
  
  if (!MyPlaneInfo(basename, &plane)) {
    printf("[GetVentricle] Calculating symmetry parameters\n");
    plane = CalculateSymmetry(iscn, (char *)basename, 0);
    WritePlaneInfo((char *)basename, plane.normal, plane.center, plane.shift);
  }
    
  vseed = VentricleMarkerScene(iscn, elems, adj, erode, plane, otsu);

  /* Debug */
  if (debug) {
    printf("Writing debug information... ");
    sprintf(filename, "%s-grad.scn", basename);
    WriteScene(grad, filename);
    sprintf(filename, "%s-bmarker.scn", basename);
    WriteScene(bseed, filename);
    sprintf(filename, "%s-vmarker.scn", basename);
    WriteScene(vseed, filename);
    sprintf(filename, "%s-int.scn", basename);
    WriteScene(iscn, filename);
    printf("done!\n");
  }
  
  W = grad->xsize;
  H = grad->ysize;
  D = grad->zsize;
  N = W*H*D;

  cost  = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  pred  = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  label = CreateScene(grad->xsize, grad->ysize, grad->zsize);

  for(i=0;i<N;i++)
    cost->data[i] = INT_MAX;

  A = Spheric(1.0);

  for(i=0;i<N;i++) {
    v.x = VoxelX(grad, i);
    v.y = VoxelY(grad, i);
    v.z = VoxelZ(grad, i);

    /* Sementes internas nos ventr�culos */
    if (vseed->data[i] != 0) {
      label->data[i] = 1;
      pred->data[i] = -1;
      cost->data[i] = 0;
      InsertSet(&S, i);
    }

    /* Sementes externas no c�rebro */
    if (bseed->data[i] != 0) {
      label->data[i] = 0;
      pred->data[i] = -1;
      cost->data[i] = 0;
      InsertSet(&S, i);
    }

    /* Sementes externas na borda da cena */
    if (v.x == 0 || v.x == W-1 || v.y == 0 || v.y == H-1 ||
	v.z == 0 || v.z == D-1) {
      label->data[i] = 0;
      pred->data[i] = -1;
      cost->data[i] = 0;
      InsertSet(&S, i);
    }
  }

  DestroyScene(&bseed);
  DestroyScene(&vseed);
  Q = CreateQueue(65536, N);

  while(S != NULL) {
    i = RemoveSet(&S);
    InsertQueue(Q,cost->data[i],i); 
  }

  while(!EmptyQueue(Q)) {
    p = RemoveQueue(Q);
    v.x = VoxelX(grad, p);
    v.y = VoxelY(grad, p);
    v.z = VoxelZ(grad, p);

    for(i=1;i<A->n;i++) {
      u.x = v.x + A->dx[i];
      u.y = v.y + A->dy[i];
      u.z = v.z + A->dz[i];

      if (ValidVoxel(grad, u.x,u.y,u.z)) {
	q = u.x + grad->tby[u.y] + grad->tbz[u.z];
	
	c1 = MAX(cost->data[p],grad->data[q]);
	if (c1 < cost->data[q]) {
	  pred->data[q] = p;
	  label->data[q] = label->data[p];
	  if (Q->L.elem[q].color == GRAY)
	    UpdateQueue(Q,q,cost->data[q],c1);
	  else
	    InsertQueue(Q,c1,q);
	  cost->data[q] = c1;
	}
      }
    }
  }

/*   sprintf(filename, "%s-ventricle.scn", basename); */
/*   WriteScene(label, filename); */
  DestroyScene(&grad);
  DestroyScene(&pred);
  DestroyScene(&cost);
  DestroySet(&S);
  DestroyAdjRel3(&A);
  DestroyQueue(&Q);
  
  return (label);
}
