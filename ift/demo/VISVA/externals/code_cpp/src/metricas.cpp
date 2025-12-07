#include "metricas.h"
#include <time.h>

char filename[100];

Image * iftDilation2D(Image *border, AdjRel *A) {
  GQueue *Q=NULL;
  int i, p, q, n, r, sz, tmp=INT_MAX;
  Pixel u, v, w;
  Image *root = NULL, *cost = NULL;

  n = border->ncols*border->nrows;
  root = CreateImage(border->ncols, border->nrows);
  cost = CreateImage(border->ncols, border->nrows);

  sz = 2*(border->ncols+border->nrows);

  Q = CreateGQueue(sz, n, cost->val);

  /* Initialization */
  for (i = 0; i < n; i++) {
    if (border->val[i] > 0) {
      cost->val[i] = 0;
      root->val[i] = i;
      InsertGQueue(&Q, i);
    }
    else
      cost->val[i] = INT_MAX;
  }

  while(!EmptyGQueue(Q)) {
    p = RemoveGQueue(Q);
    u.x = p % cost->ncols;
    u.y = p / cost->ncols;

    /* Resgata coordenadas da raiz de p */
    r = root->val[p];
    w.x = r % root->ncols;
    w.y = r / root->ncols;

    for (i = 1; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(cost, v.x, v.y)) {
	q = v.x + cost->tbrow[v.y];
	if (cost->val[q] > cost->val[p]) {
          tmp = QD(v.x, w.x) + QD(v.y, w.y);
          if (tmp < cost->val[q]) {
            if (cost->val[q] != INT_MAX) {
              RemoveGQueueElem(Q, q);
            }
            cost->val[q] = tmp;
            root->val[q] = root->val[p];
            InsertGQueue(&Q, q);
          }
        }
      }
    }
  }
  
  DestroyGQueue(&Q);
  DestroyImage(&root);

  return (cost);
    
}


Polynom *MSFractal2D(Image *bin, int maxdist, int degree, double lower, 
		   double higher, int reg, double from, double to)
{
  Curve *hist=NULL,*haux=NULL,*ahist=NULL, *aux_ahist=NULL,*loglog=NULL;
  AdjRel *A=NULL;
  Image *mbb=NULL,*nbin=NULL, *cost = NULL;
  Polynom *P=NULL,*D=NULL;
  int n,i,j,maxcost=maxdist*maxdist;
    
  /* bin -> imagem de borda */
  mbb  = MBB(bin);
  if (mbb == NULL)
    return NULL;

  nbin = AddFrame(mbb,maxdist,0);
  DestroyImage(&mbb);

  /* Compute Euclidean IFT */
  A = Circular(1.5);
  cost = iftDilation2D(nbin, A);
  DestroyAdjRel(&A);

  /* Compute MS Fractal */
  hist = Histogram(cost);

  /* Compute non-zero points */

  n = 0;
  for (i=1; i < maxcost; i++)
    if (hist->Y[i] != 0)
      n++;

  haux = CreateCurve(n);
  j=0;
  for (i=1; i < maxcost; i++)
    if (hist->Y[i] != 0) {
      haux->X[j] = log(sqrt((double)i));
      haux->Y[j] = hist->Y[i];
      j++;
    }
  
  /* Accumulate values */
  ahist = CreateCurve(n);
  ahist->X[0] = haux->X[0];
  ahist->Y[0] = haux->Y[0];
  for (i=1; i < n; i++) {
    ahist->X[i] = haux->X[i];
    ahist->Y[i] = ahist->Y[i-1] + haux->Y[i];
  }

  /* Compute log(Y) */
  for (i=0; i < n; i++)
    ahist->Y[i] = log((double)ahist->Y[i]);
  
  j=0;
 
  for (i=0; i < n; i++)
    if ((ahist->X[i]>from)&&((ahist->X[i]<to)))
      j++;
  
  aux_ahist = CreateCurve(j);
  
  j=0;
  for (i=0; i < n; i++)
    if ((ahist->X[i]>from)&&((ahist->X[i]<to))){
      aux_ahist->X[j] = ahist->X[i];
      aux_ahist->Y[j] = ahist->Y[i];
      j++;
    }
  
  
  /* Compute Regression */
  P = Regression(aux_ahist, degree);
  
  /* Print loglog curve */
  if (reg){
    loglog = SamplePolynom(P, lower, higher, 100);
    WriteCurve(ahist, (char *)"experimental2");
    WriteCurve(loglog, (char *)"regression2");
  }
  
  /* Compute Fractal Curve */
  D = DerivPolynom(P);  

  DestroyCurve(&hist);
  DestroyCurve(&haux);
  DestroyCurve(&ahist);
  DestroyCurve(&aux_ahist);
  DestroyCurve(&loglog);
  DestroyPolynom(&P);
  DestroyImage(&cost);
  DestroyImage(&nbin);
  return(D);
}


Curve *ContourMSFractal2D(Image *in, int nbins)
{
  Image *cont = NULL;
  Curve *descriptor = NULL;
  Polynom *P;
  double lower = 1.0;
  double higher = 5.0;
  int maxdist = 256;
  int degree = 10;

  cont = LabelContour(in);
  P = MSFractal2D(cont, maxdist, degree, lower, higher, 0, 0.0, 6.0);
  if (P != NULL)
    descriptor = PolynomToFractalCurve(P, lower, higher, nbins);
  
  DestroyPolynom(&P);
  DestroyImage(&cont);
  return (descriptor);
}



Scene * GetHemisphere(Scene *bin, int side, DVoxel normal,
                      DVoxel center, double xshift) {

  Scene *hemi = NULL;
  DVoxel v;
  int i, j, k, p;
  
  hemi = CreateScene(bin->xsize, bin->ysize, bin->zsize);
  for (i = 0; i < bin->xsize; i++) {
    for (j = 0; j < bin->ysize; j++) {
      for (k = 0; k < bin->zsize; k++) {
        v.x = (double)i - center.x;
        v.y = (double)j - center.y;
        v.z = (double)k - center.z;
        if ((side*InternalProduct(normal, v)) > 0.0) {
          p = i + (int)xshift + hemi->tby[j] + hemi->tbz[k];
          hemi->data[p] = bin->data[p];
        }
      }
    }
  }
  
  return (hemi);
  
}

Hemispheres * GetHemispheres(Scene *bin, DVoxel normal, DVoxel center,
                             double xshift) {

  Hemispheres *brain = NULL;
  DVoxel v;
  int i, j, k, p;

  brain = (Hemispheres *)malloc(sizeof(Hemispheres));
  brain->left = CreateScene(bin->xsize, bin->ysize, bin->zsize);
  brain->right = CreateScene(bin->xsize, bin->ysize, bin->zsize);
  
  for (i = 0; i < bin->xsize; i++) {
    for (j = 0; j < bin->ysize; j++) {
      for (k = 0; k < bin->zsize; k++) {
        v.x = (double)i - center.x;
        v.y = (double)j - center.y;
        v.z = (double)k - center.z;
        p = i + (int)xshift + bin->tby[j] + bin->tbz[k];
        if (InternalProduct(normal, v) > 0.0)
          brain->right->data[p] = bin->data[p];
        else if (InternalProduct(normal, v) < 0.0)
          brain->left->data[p] = bin->data[p];
      }
    }
  }

  return (brain);
  
}


void SumMetric(Scene *bin) {

  DVoxel n, c, v, vs, rfx;
  int l = 0, r = 0, t = 0, diff = 0;
  int i, j, k, vl = 0, vl_rfx = 0;
  double shift;

  if ((ReadPlaneInfo(filename, &n, &c, &shift)) == NIL) {
    printf("[SumMetric] Impossible to calculate scene metric\n");
    return;
  }

  for (i = 0; i < bin->xsize; i++) {
    for (j = 0; j < bin->ysize; j++) {
      for (k = 0; k < bin->zsize; k++) {
        v.x = (double)i;
        v.y = (double)j;
        v.z = (double)k;
        vs.x = (double)i - c.x;
        vs.y = (double)j - c.y;
        vs.z = (double)k - c.z;
        VectorReflection(v, c, n, &rfx);
        if (ValidVoxel(bin, (int)rfx.x, (int)rfx.y, (int)rfx.z)) {
          vl = bin->data[i + bin->tby[j] + bin->tbz[k]];
          vl_rfx = bin->data[ROUND(rfx.x) + bin->tby[ROUND(rfx.y)] + bin->tbz[ROUND(rfx.z)]];
          diff = abs(vl - vl_rfx);
          if (InternalProduct(n, vs) < 0.0) {
            l += diff;
          }
          else if (InternalProduct(n, vs) > 0.0) {
            r += diff;
          }
          t += vl;
        }
      }
    }
  }
  
  printf("\n[SumMetric]\n");
  printf("LEFT/RIGHT: %d/%d == %f\n", l, r, (float)l/(float)r);
  printf("TOTAL: %d\n", t);

  printf("(LEFT+RIGHT)/(2*TOTAL) == %f\n", ((float)l+(float)r)/(2*(float)t));

  return;

}

void HemispheresVolume(Scene *scn) {

  DVoxel n, c, xv;
  int left_vol = 0, right_vol = 0;
  int i, j, k;
  double shift;
  Scene *vol = NULL;
  int Imax=MaximumValue3(scn);
  char fname[100];

  if ((ReadPlaneInfo(filename, &n, &c, &shift)) == NIL) {
    printf("[HemispheresVolume] Impossible to calculate scene volumes\n");
    return;
  }

  vol = CopyScene(scn);

  for (i = 0; i < scn->xsize; i++) {
    for (j = 0; j < scn->ysize; j++) {
      for (k = 0; k < scn->zsize; k++) {
        xv.x = (double)i - c.x;
        xv.y = (double)j - c.y;
        xv.z = (double)k - c.z;
        if (scn->data[i + scn->tby[j] + scn->tbz[k]] > 0.0) {
          if (InternalProduct(n, xv) < 0.0) {
            left_vol++;
            vol->data[i + vol->tby[j] + vol->tbz[k]] = Imax;
          }
          else if (InternalProduct(n, xv) > 0.0) {
            right_vol++;
            vol->data[i + vol->tby[j] + vol->tbz[k]] = Imax/2;
          }
        }
      }
    }
  }
  
  sprintf(fname, "%s-volume.scn", filename);
  WriteScene(vol, fname);
  DestroyScene(&vol);

  printf("\n[HemispheresVolume]\n");
  printf("LEFT VOLUME: %d\n", left_vol);
  printf("RIGHT VOLUME: %d\n", right_vol);

  printf("RATIO (L/R): %.4f\n", (float)left_vol/right_vol);

}


Scene * iftDilation3D(Scene *border, AdjRel3 *A) {
  GQueue *Q=NULL;
  int i, p, q, n, s, r, tmp=INT_MAX;
  Voxel u, v, w;
  Scene *root = NULL, *cost = NULL;

  s = border->xsize*border->ysize;
  n = border->xsize*border->ysize*border->zsize;
  root = CreateScene(border->xsize, border->ysize, border->zsize);
  cost = CreateScene(border->xsize, border->ysize, border->zsize);

  Q = CreateGQueue(1024, n, cost->data);

  /* Initialization */
  for (i = 0; i < n; i++) {
    if (border->data[i] > 0) {
      cost->data[i] = 0;
      root->data[i] = i;
      InsertGQueue(&Q, i);
    }
    else
      cost->data[i] = INT_MAX;
  }

  while(!EmptyGQueue(Q)) {
    p = RemoveGQueue(Q);
    u.z = p / s;
    i = (p % s);
    u.x = i % cost->xsize;
    u.y = i / cost->xsize;

    /* Resgata coordenadas da raiz de p */
    r = root->data[p];
    w.x = (r % s) % root->xsize;
    w.y = (r % s) / root->xsize;
    w.z = r / s;

    for (i = 1; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(cost, v.x, v.y, v.z)) {
	q = v.x + cost->tby[v.y] + cost->tbz[v.z];
	if (cost->data[q] > cost->data[p]) {
          tmp = QD(v.x, w.x) + QD(v.y, w.y) + QD(v.z, w.z);
          if (tmp < cost->data[q]) {
            if (cost->data[q] != INT_MAX) {
              RemoveGQueueElem(Q, q);
            }
            cost->data[q] = tmp;
            root->data[q] = root->data[p];
            InsertGQueue(&Q, q);
          }
        }
      }
    }
  }
  
  DestroyGQueue(&Q);
  DestroyScene(&root);

  return (cost);
    
}


Polynom *MSFractal3D(Scene *border, int maxdist, int degree, double lower, 
                     double higher, int reg, double from, double to)
{
  Curve *hist=NULL, *ahist=NULL, *haux=NULL, *aux_ahist=NULL, *loglog=NULL;
  AdjRel3 *A = NULL;
  Scene *mbb = NULL,*nborder = NULL, *cost = NULL;
  Polynom *P = NULL, *D = NULL;
  int n, i, j, maxcost = maxdist*maxdist;
  char name[100];

  /*  minimum bounding box */
  mbb  = MBB3(border);
  nborder = AddFrame3(mbb, maxdist, 0);
  DestroyScene(&mbb);

  /* Compute TDE (Euclidean Distance Transform) */
  A = Spheric(1.5);
  n = nborder->xsize*nborder->ysize*nborder->zsize;
  cost = iftDilation3D(nborder, A);
  
  DestroyAdjRel3(&A);
  DestroyScene(&nborder);

  /* Compute MS Fractal */
  hist = Histogram3(cost);

  /* Compute non-zero points */
  n = 0;
  for (i=1; i < maxcost; i++)
    if (hist->Y[i] != 0)
      n++;

  haux = CreateCurve(n);
  ahist = CreateCurve(n);
  j = 0;
  for (i=1; i < maxcost; i++)
    if (hist->Y[i] != 0) {
      haux->X[j] = log(sqrt((double)i));
      haux->Y[j] = hist->Y[i];
      j++;
    }
  
  /* Accumulate values */
  ahist->X[0] = haux->X[0];
  ahist->Y[0] = haux->Y[0];
  for (i=1; i < n; i++) {
    ahist->X[i] = haux->X[i];
    ahist->Y[i] = ahist->Y[i-1] + haux->Y[i];
  }

  /* Compute log(Y) */
  for (i=0; i < n; i++)
    ahist->Y[i] = log((double)ahist->Y[i]);
  
  j=0;
  for (i=0; i < n; i++)
    if ((ahist->X[i] > from) && (ahist->X[i] < to))
      j++;
  
  aux_ahist = CreateCurve(j);
  
  j=0;
  for (i=0; i < n; i++)
    if ((ahist->X[i] > from) && (ahist->X[i] < to)){
      aux_ahist->X[j] = ahist->X[i];
      aux_ahist->Y[j] = ahist->Y[i];
      j++;
    }
  
  /* Compute Regression */
  P = Regression(aux_ahist, degree);
  
  /* Print loglog curve */
  if (reg) {
    loglog = SamplePolynom(P, lower, higher, 100);
    sprintf(name, "%s-experimental", filename);
    WriteCurve(ahist, name);
    sprintf(name, "%s-regression", filename);
    WriteCurve(loglog, name);
  }
  
  /* Compute Fractal Curve */
  D = DerivPolynom(P);
  
  DestroyCurve(&hist);
  DestroyCurve(&ahist);
  DestroyCurve(&haux);
  DestroyCurve(&aux_ahist);
  DestroyCurve(&loglog);
  DestroyAdjRel3(&A);
  DestroyScene(&mbb);
  DestroyScene(&nborder);
  DestroyScene(&cost);
  DestroyPolynom(&P);
  
  return(D);
}


Curve * ContourMSFractal3D(Scene *bin) {
  
  Scene *border = NULL;
  Curve *descriptor = NULL;
  AdjRel3 *A=Spheric(1.5);
  Polynom *P = NULL;
  int nbins = 100; /* Quantização da curva */
  int degree = 10; /* Grau do Polinômio */
  int maxdist = 96;
  double from = 0.0;
  double lower = 1.0;
  double to = 4.0;
  double higher = 4.0;
  
  border = GetBorder3(bin, A);
  P = MSFractal3D(border, maxdist, degree, lower, higher, 1, from, to);
  descriptor = PolynomToFractalCurve(P, lower, higher, nbins);
  
  DestroyScene(&border);
  DestroyAdjRel3(&A);
  DestroyPolynom(&P);
  
  return (descriptor);
  
}

Scene *BinarizeVolume(Scene *scn) {
  
  Scene *bin = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  int i;
  int n = scn->xsize*scn->ysize*scn->zsize;
  
  for (i = 0; i < n; i++)
    if (scn->data[i] != 0)
      bin->data[i] = 1;

  return (bin);
  
}

Image *BinarizeImage(Image *img) {
  
  Image *bin = CreateImage(img->ncols, img->nrows);
  int i;
  int n = img->ncols*img->nrows;
  
  for (i = 0; i < n; i++)
    if (img->val[i] != 0)
      bin->val[i] = 1;

  return (bin);
  
}

Curve * DiffCurves(Curve *curve1, Curve *curve2, double factor) {

  int i;
  Curve *curve = NULL;

  if (curve1->n != curve2->n)
    return NULL;

  curve = CreateCurve(curve1->n);
  for (i = 0; i < curve1->n; i++) {
    curve->Y[i] = fabs(curve2->Y[i] - curve1->Y[i]) + factor;
    curve->X[i] = curve1->X[i];
  }

  return (curve);

}

Curve * StandardDeviationCurve(Curve **curves, Curve *mcurve, int ncurves) {
  /*
    Desvio Padrao == SQRT(Variancia)
    Variancia == (Sum(x_i^2) - n*x_m^2)/(n - 1)
   */
  
  Curve *sdcurve = NULL;
  int i, j;
  double y, ym, s;
  
  sdcurve = CreateCurve(mcurve->n);
  for (i = 0; i < mcurve->n; i++) {
    s = 0.0;
    for (j = 0; j < ncurves; j++) {
      y = curves[j]->Y[i];
      s += y*y;
    }
    ym = mcurve->Y[i];
    sdcurve->Y[i] = sqrt((s - ncurves*ym*ym)/(ncurves - 1));
    sdcurve->X[i] = mcurve->X[i];
  }

  return (sdcurve);

}


Curve *MedianCurve(Curve **curves, int ncurves) {

  Curve *mcurve = NULL;
  int i, j, n;
  double sum;

  mcurve = CreateCurve(ncurves);
  n = curves[0]->n;

  for (i = 0; i < n; i++)
    mcurve->X[i] = curves[0]->X[i];
     

  for (i = 0; i < n; i++) {
    sum = 0.0;
    for (j = 0; j < ncurves; j++) {
      sum += curves[j]->Y[i];
    }
    mcurve->Y[i] = sum/ncurves;
  }

  return (mcurve);

}


/*
  Calcula a curva fractal de cada fatia da cena
  e retorna a curva média
 */
Curve * HemispheresMedianCurve(Scene *bin) {

  Scene *mbb3 = NULL;
  Image *img = NULL;
  Curve **curve = NULL, *mcurve = NULL;
  int i, j, ncurves, tmp;
  int nbins = 100;
  double sum;
  
  mbb3 = MBB3(bin);
  
  ncurves = mbb3->zsize;

  curve = (Curve **)malloc(ncurves*sizeof(Curve *));
  for (i = 0; i < ncurves; i++) {
    img = GetSlice(mbb3, i);
    curve[i] = ContourMSFractal2D(img, nbins);
    DestroyImage(&img);
  }

  /* Média das curvas */
  mcurve = CreateCurve(nbins);
  for (i = 0; i < nbins; i++) {
    sum = 0;
    tmp = ncurves;
    for (j = 0; j < ncurves; j++) {
      if (curve[j] != NULL)
        sum += curve[j]->Y[i];
      else tmp--;
    }
    mcurve->Y[i] = sum/tmp;
    mcurve->X[i] = curve[1]->X[i];
  }

  /* Liberação de memória */
  for (i = 0; i < mbb3->zsize; i++)
    DestroyCurve(&curve[i]);

  free(curve);
  DestroyScene(&mbb3);

  return (mcurve);
}


/* Curve * Fractal2D(char *basename) { */
Curve * Fractal2D(char *basename, Scene *scn) {

  Scene *iscn = NULL, *bin = NULL, *lscn = NULL;
  Hemispheres *brain = NULL;
  Curve *lcurve = NULL, *rcurve = NULL, *dcurve = NULL;
  DVoxel normal, center;/* , x, z, n; */
  double shift;/* , costheta, sintheta; */
/*   char *destdir = "curvas"; */
/*   char *fname = NULL; */

/*   sprintf(filename, "%s.scn", basename); */
  lscn = CopyScene(scn);
  iscn = LinearInterp(lscn, 1.0, 1.0, 1.0);

  if ((ReadPlaneInfo(basename, &normal, &center, &shift)) == NIL) {
    printf("[Fractal2D] Impossible to get plane info\n");
    return NULL;
  }

  /* Rotaciona a normal ao plano */
  /*   z.x = scn->xsize; */
  /*   z.y = z.z = 0.0; */
  
  /*   x = NormalVector(normal, z); */
  
  /*   costheta = GetCosine(normal, z); */
  /*   sintheta = GetSine(costheta); */
  
  /*   n.x = normal.x*iscn->xsize; */
  /*   n.y = normal.y*iscn->xsize; */
  /*   n.z = normal.z*iscn->xsize; */
  
  /*   printf("Normal antes %f %f %f\n", n.x, n.y, n.z); */
  /*   n = RotateArbitrary(n, center, x, costheta, sintheta); */
  /*   printf("Normal depois %f %f %f\n", n.x, n.y, n.z); */
  
  /* Rotaciona a cena para assegurar que os cortes serão
     paralelos ao plano de simetria */
  DestroyScene(&lscn);
  lscn = RotateScene(iscn, normal, center, shift, 0);

  bin = BinarizeVolume(lscn);

  normal.x = bin->xsize;
  normal.y = 0;
  normal.z = 0;

  brain = GetHemispheres(bin, normal, center, shift);

  lcurve = HemispheresMedianCurve(brain->left);
  rcurve = HemispheresMedianCurve(brain->right);

  /* O parâmetro 0.7 é aplicado somente para melhor visualizar a curva diff */
  /* TODO: Não esquecer de remover o parâmetro 0.7 */
  dcurve = DiffCurves(lcurve, rcurve, 0.7);

/*   fname = strrchr(basename, '/'); */
/*   sprintf(filename, "%s-lcurve.txt", fname); */
/*   WriteCurve(lcurve, filename); */
/*   sprintf(filename, "%s-rcurve.txt", fname); */
/*   WriteCurve(rcurve, filename); */
/*   sprintf(filename, "%s-dcurve.txt", fname); */
/*   WriteCurve(dcurve, filename); */

  DestroyCurve(&lcurve);
  DestroyCurve(&rcurve);
  /* DestroyCurve(&dcurve); */
  DestroyScene(&lscn);
  DestroyScene(&iscn);
  DestroyScene(&bin);
  DestroyScene(&brain->left);
  DestroyScene(&brain->right);
  free(brain);

  return (dcurve);

}


void Fractal3D(char *basename) {

  Scene *iscn = NULL, *scn = NULL, *bin = NULL, *right = NULL, *left = NULL;
  Curve *curve = NULL;
  int i;
  FILE *fp=NULL;
  DVoxel n, c;
  double shift;
  
  sprintf(basename, "%s.scn", basename);
  scn = ReadScene(basename);

  iscn = LinearInterp(scn, 1.0, 1.0, 1.0);
  DestroyScene(&scn);

  if ((ReadPlaneInfo(basename, &n, &c, &shift)) == NIL) {
    printf("[main] Impossible to get plane info\n");
    return;
  }

  sprintf(filename, "%s", basename);
  
  bin = BinarizeVolume(iscn);

  left = GetHemisphere(bin, RIGHT, n, c, shift);

  curve = ContourMSFractal3D(left);
  DestroyScene(&iscn);
  DestroyScene(&bin);
  DestroyScene(&left);
  DestroyScene(&right);

  if (curve != NULL) {
    sprintf(filename, "%s-curve3d", basename);
    fp = fopen(filename, "wb");
    fprintf(fp, "%d\n", curve->n);
    for (i = 0; i < curve->n; i++)
      fprintf(fp, "%3.2f %3.2f\n", curve->X[i], curve->Y[i]);

    fclose(fp);
    DestroyCurve(&curve);
    DestroyScene(&iscn);
    DestroyScene(&scn);
    DestroyScene(&bin);
    DestroyScene(&right);
    DestroyScene(&left);

  }

}


/* int main(int argc, char **argv) { */

/*   char *dim = "3D"; */

/*   if ((argc != 2) && (argc != 3)) { */
/*     printf("Uso: %s <cena-sem-extensao> <dimension>[2D(default), 3D]\n", argv[0]); */
/*     return (-1);     */
/*   } */

/*   if ((argc == 3) && (strcmp(dim, argv[2]) == 0)) */
/*     printf("Cálculos 3D\n"); */
/*   else { */
/*     Fractal2D(argv[1]); */
/*   } */

/*   printf("Tempo de execução: %.2f segundos.\n", (float)clock()/CLOCKS_PER_SEC); */

/*   return 0; */
  
/* } */
