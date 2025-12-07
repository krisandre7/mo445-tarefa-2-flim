#include "simetria.h"
#include <time.h>

void Normalise(DVoxel *v) {
  double t;

  t = sqrt(v->x*v->x + v->y*v->y + v->z*v->z);

  v->x /= t;
  v->y /= t;
  v->z /= t;
  
}

/* int dvsize; */

DVoxel RotateArbitrary(DVoxel p, DVoxel c, DVoxel n,
                       double costheta, double sintheta) {
  /*
    Rotaciona um ponto por um ângulo arbitrário
    p: ponto origem
    c: o centro da cena
    n: a normal
  */

  DVoxel q1, q2, u;
  double d;

  /* Step 1 */
  q1.x = p.x - c.x;
  q1.y = p.y - c.y;
  q1.z = p.z - c.z;

  u.x = n.x - c.x;
  u.y = n.y - c.y;
  u.z = n.z - c.z;

  Normalise(&u);

  d = sqrt(u.y*u.y + u.z*u.z);

  /* Step 2 */
  if (d != 0) {
    q2.x = q1.x;
    q2.y = q1.y * u.z / d - q1.z * u.y / d;
    q2.z = q1.y * u.y / d + q1.z * u.z / d;
  } else {
    q2.x = q1.x;
    q2.y = q1.y;
    q2.z = q1.z;
  }

  /* Step 3 */
  q1.x = q2.x*d - q2.z*u.x;
  q1.y = q2.y;
  q1.z = q2.x*u.x + q2.z*d;

  /* Step 4 */
  q2.x = q1.x*costheta + q1.y*sintheta;
  q2.y = (-1)*q1.x*sintheta + q1.y*costheta;
  q2.z = q1.z;

  /* Step 5 */
  q1.x = q2.x*d + q2.z*u.x;
  q1.y = q2.y;
  q1.z = (-1)*q2.x*u.x + q2.z*d;

  /* Step 6 */
  if (d != 0) {
    q2.x = q1.x;
    q2.y = q1.y*u.z/d + q1.z*u.y/d;
    q2.z = (-1)*q1.y*u.y/d + q1.z*u.z/d;
  } else {
    q2.x = q1.x;
    q2.y = q1.y;
    q2.z = q1.z;    
  }

  /* Step 7 */
  q1.x = q2.x + c.x;
  q1.y = q2.y + c.y;
  q1.z = q2.z + c.z;

  return q1;

}


void VectorReflection(DVoxel p1, DVoxel center, DVoxel n, DVoxel *p) {
  /*
    Faz a reflexão do vetor p1 em relação a n
    p1: o ponto que será refletido
    center: o centro de gravidade do objeto
    n: o vetor normal ao plano de corte
    p: o vetor refletido

    Fórmula de Reflexão:
       p = center + v - 2(v*n)*n
    Onde v = p1 - center
  */

   double w;
   DVoxel v, temp;

  v.x = p1.x - center.x;
  v.y = p1.y - center.y;
  v.z = p1.z - center.z;
  
  w = (p1.x-center.x)*n.x + (p1.y-center.y)*n.y + (p1.z-center.z)*n.z;

  temp.x = center.x + v.x - 2*w*n.x;
  temp.y = center.y + v.y - 2*w*n.y;
  temp.z = center.z + v.z - 2*w*n.z;

  /* Arredondamento */
  p->x = (int)floor(temp.x+0.5);
  p->y = (int)floor(temp.y+0.5);
  p->z = (int)floor(temp.z+0.5);

  return;
  
}

void WritePlaneInfo(char *filename, DVoxel n, DVoxel center, double x_shift) {
  /*
    Armazena as coordenadas do vetor normal e do centro de gravidade do objeto
    num arquivo binário.
    - filename: o nome do arquivo base, sem extensão
    - n: o vetor normal
    - center: o centro de gravidade do objeto
    - x_shift: o deslocamento do eixo x
  */

  char fname[100] = "";
  FILE *fp;

  sprintf(fname, "%s.info", filename);
  if ((fp = fopen(fname, "wb")) == NULL) {
    printf("[WritePlaneInfo] Cannot open file %s for writing\n", fname);
    return;
  }

  fwrite(&n, sizeof(DVoxel), 1, fp);
  fwrite(&center, sizeof(DVoxel), 1, fp);
  fwrite(&x_shift, sizeof(double), 1, fp);

  fclose(fp);

}
  
int ReadPlaneInfo(char *filename, DVoxel *n,
                  DVoxel *center, double *x_shift) {
  /*
    Resgata as coordenadas do vetor normal e do centro de gravidade do objeto
    de um arquivo binário.
    - filename: o nome do arquivo base, sem extensão
    - n: o vetor normal
    - center: o centro de gravidade do objeto
    - x_shift: o deslocamento do eixo x
  */

  char fname[100] = "";
  FILE *fp;

  sprintf(fname, "%s.info", filename);
  if ((fp = fopen(fname, "rb")) == NULL) {
    printf("[ReadPlaneInfo] Cannot open file %s for reading\n", fname);
    return NIL;
  }

  fread(n, 3*sizeof(double), 1, fp);
  fread(center, 3*sizeof(double), 1, fp);
  fread(x_shift, sizeof(double), 1, fp);

  fclose(fp);

  return (0);

}


double GetCosine(DVoxel u, DVoxel v) {
  /*
    Retorna o coseno do ângulo entre dois vetores
    Fórmula:
               u*v
    cos(T) = ---------
             |u|*|v|
  */

  double n, d;

  n = u.x*v.x + u.y*v.y + u.z*v.z;

  d = sqrt(u.x*u.x + u.y*u.y + u.z*u.z) *
    sqrt(v.x*v.x + v.y*v.y + v.z*v.z);

  return (n/d); /* cos(theta) */

}

double GetSine(double cosine) {
  /* 
     Retorna o seno de um ângulo dados seu cosseno
     sen^2(theta) + cos^2(theta) = 1
  */
  return sqrt(1 - cosine*cosine);

}

double GetAngle(double cosine) {

  return 180*acos(cosine)/PI;

}


int CompareSlices(Slice *slc1, Slice *slc2) {
  /*
    Função de comparação para o quick sort
   */

  if (slc1->intensity < slc2->intensity)
    return -1;
  else if (slc1->intensity > slc2->intensity)
    return 1;
  
  return 0;
  
}

void RotatePitch(DVoxel n, double degree, DVoxel *m) {
  /*
    Rotaciona no eixo 'x'
    Matriz de rotação:
    | 1.0  0.0     0.0    |
    | 0.0  cos(T)  sin(T) |
    | 0.0 -sin(T)  cos(T) |
  */
  double matrix[3][3];
  double rad = degree*PI/180.0;
  
  matrix[0][0] = 1.0;
  matrix[0][1] = matrix[0][2] = matrix[1][0] = matrix[2][0] = 0.0;
  matrix[1][1] = matrix[2][2] = cos(rad);
  matrix[2][1] = -(1.0)*sin(rad);
  matrix[1][2] = sin(rad);

  m->x = n.x*matrix[0][0] + n.y*matrix[1][0] + n.z*matrix[2][0];
  m->y = n.x*matrix[0][1] + n.y*matrix[1][1] + n.z*matrix[2][1];
  m->z = n.x*matrix[0][2] + n.y*matrix[1][2] + n.z*matrix[2][2];

}

void RotateRoll(DVoxel n, double degree, DVoxel *m) {
  /*
    Rotaciona no eixo 'y'
    Matriz de rotação:
    | cos(T)  0.0 -sin(T) |
    |   0.0   1.0  0.0    |
    | sin(T)  0.0  cos(T) |
  */
  double matrix[3][3];
  double rad = degree*PI/180.0;
  
  matrix[0][0] = matrix[2][2] = cos(rad);
  matrix[2][0] = sin(rad);
  matrix[0][2] = (-1)*sin(rad);
  matrix[1][1] = 1.0;
  matrix[0][1] = matrix[1][0] = matrix[1][2] = matrix[2][1] = 0.0;

  m->x = n.x*matrix[0][0] + n.y*matrix[1][0] + n.z*matrix[2][0];
  m->y = n.x*matrix[0][1] + n.y*matrix[1][1] + n.z*matrix[2][1];
  m->z = n.x*matrix[0][2] + n.y*matrix[1][2] + n.z*matrix[2][2];

}

void RotateYaw(DVoxel n, double degree, DVoxel *m) {
  /*
    Rotaciona no eixo 'z'
    Matriz de rotação:
    |  cos(T)  sin(T)  0.0 |
    | -sin(T)  cos(T)  0.0 |
    |   0.0     0.0    1.0 |
  */
  double matrix[3][3];
  double rad = degree*PI/180.0;
  
  matrix[0][0] = matrix[1][1] = cos(rad);
  matrix[0][1] = sin(rad);
  matrix[1][0] = (-1)*sin(rad);
  matrix[2][1] = matrix[2][0] = matrix[0][2] = matrix[1][2] = 0.0;
  matrix[2][2] = 1.0;

  m->x = n.x*matrix[0][0] + n.y*matrix[1][0] + n.z*matrix[2][0];
  m->y = n.x*matrix[0][1] + n.y*matrix[1][1] + n.z*matrix[2][1];
  m->z = n.x*matrix[0][2] + n.y*matrix[1][2] + n.z*matrix[2][2];

}

void MinAndMax(double n1, double n2, double n3, double *max1, double *max2) {
  /*
    Retorna o menor e o maior valor
  */
  if (((n1 >= n2) && (n1 <= n3)) || ((n1 <= n2) && (n1 >= n3))) {
    *max1 = n2;
    *max2 = n3;
  }
  else if (((n2 >= n1) && (n2 <= n3)) || ((n2 <= n1) && (n2 >= n3))) {
    *max1 = n1;
    *max2 = n3;
  }
  else if (((n3 >= n1) && (n3 <= n2)) || ((n3 <= n1) && (n3 >= n2))) {
    *max1 = n1;
    *max2 = n2;
  }
  
}

void MinTwo(double n1, double n2, double n3, double * max1, double * max2) {
  /*
    Retorna os dois maiores valores
  */
  if ((n1 >= n2) && (n1 >= n3)) {
    *max1 = n2;
    *max2 = n3;
  }
  else if ((n2 >= n1) && (n2 >= n3)) {
    *max1 = n1;
    *max2 = n3;
  }
  else if ((n3 >= n1) && (n3 >= n2)) {
    *max1 = n1;
    *max2 = n2;
  }

}

void MaxTwo(double n1, double n2, double n3, double * max1, double * max2) {
  /*
    Retorna os dois maiores valores
  */
  if ((n1 <= n2) && (n1 <= n3)) {
    *max1 = n2;
    *max2 = n3;
  }
  else if ((n2 <= n1) && (n2 <= n3)) {
    *max1 = n1;
    *max2 = n3;
  }
  else if ((n3 <= n1) && (n3 <= n2)) {
    *max1 = n1;
    *max2 = n2;
  }

}

void Median(double n1, double n2, double n3, double *m) {

  if ((n1 >= n2) && (n1 <= n3))
    *m = n1;
  else if ((n2 >= n1) && (n2 <= n2))
    *m = n2;
  else if ((n3 >= n2) && (n3 <= n1))
    *m = n3;
  
}

double InternalProduct(DVoxel u, DVoxel v) {
  /*
   Retorna o produto interno (produto escalar) de dois vetores
  */
  return u.x*v.x + u.y*v.y + u.z*v.z;
}

Scene * GetPlane(Scene *scn, DVoxel n, DVoxel p, int shift) {
  /*
    scn: a cena de onde será extraído o plano
    n: vetor normal ao plano
    p: o centro de gravidade da cena
    shift: deslocamento no eixo 'x'
  */
  int i, j, k, x, y, z;
  int Imax=MaximumValue3(scn);
  DVoxel xv;
  Scene *plane = NULL;
  
  x = scn->xsize;
  y = scn->ysize;
  z = scn->zsize;

  plane = CopyScene(scn);

  for (i = 0; i < x; i++) {
    for (j = 0; j < y; j++) {
      for (k = 0; k < z; k++) {
        xv.x = (double)i - p.x;
        xv.y = (double)j - p.y;
        xv.z = (double)k - p.z;
        if (abs(InternalProduct(n, xv)) == 0.0) {
          plane->data[i + shift + plane->tby[j] + plane->tbz[k]] = Imax;
        }
      }
    }
  }
  
  return(plane);

}


DVoxel * GetFirstPlaneIntensity(Scene *scn, DVoxel n, DVoxel c) {
  /*
    scn: a cena de onde serão extraídas os brilhos
    n: vetor normal ao plano
    c: o centro de gravidade da cena
  */
  int i, j, k, I=0, pos, nvoxels, m;
  DVoxel xv, *ret, *v;

  nvoxels = scn->xsize*scn->ysize*scn->zsize;
  v = (DVoxel *)malloc(nvoxels*sizeof(DVoxel));

  for (i = 0; i < nvoxels; i++) {
    v[i].x = -1;
  }

  m = 0;
  for (i = 0; i < scn->xsize; i++) {
    for (j = 0; j < scn->ysize; j++) {
      for (k = 0; k < scn->zsize; k++) {
        xv.x = (double)i - c.x;
        xv.y = (double)j - c.y;
        xv.z = (double)k - c.z;
        pos = i + scn->tby[j] + scn->tbz[k];
        if ((scn->data[pos] != 0.0) && (abs(InternalProduct(n, xv)) == 0.0)) {
          m++;
          v[i].x = i;
          v[i].y = j;
          v[i].z = k;
          I += scn->data[pos];
        }
      }
    }
  }

  ret = (DVoxel *)malloc(m*sizeof(DVoxel));

  j = 0;
  for (i = 0; i < nvoxels; i++)
    if (v[i].x != -1) {
      ret[j] = v[i];
      j++;
    }

  free(v); v = 0;
  /* dvsize = m; */

  return(ret);

}

/*
  -*-*-*-*- TODO -*-*-*-*-
  Otimizar passando uma lista de coordenadas de voxel ao invés de
  limitar o escopo da busca
*/

int GetPlaneIntensity(Scene *scn, DVoxel n, DVoxel c, int shift) {
  /*
    scn: a cena de onde serão extraídas os brilhos
    n: vetor normal ao plano
    p: o centro de gravidade da cena
    shift: deslocamento no eixo 'x'
  */
  int i, j, k, x, I=0, pos;
  DVoxel xv;

  x = scn->xsize;

  for (i = 2*x/5; i < 3*x/5; i++) {
    for (j = c.y - 80; j < c.y + 80; j++) {
      for (k = c.z - 20; k < c.z + 20; k++) {
        xv.x = (double)i - c.x;
        xv.y = (double)j - c.y;
        xv.z = (double)k - c.z;
        pos = i + shift + scn->tby[j] + scn->tbz[k];
        if ((abs(InternalProduct(n, xv)) == 0.0) && (scn->data[pos] != 0.0)) {
          I += scn->data[pos];
        }
      }
    }
  }
  
  return(I);

}

double VectorModule(DVoxel v) {
  /*
    Módulo de um vetor v (x, y, z):
    |v| == sqrt(vx*vx + vy*vy + vz*vz)
  */
  return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);

}


void VectorVersor(DVoxel *u) {
  /*
    O versor de um vetor é o vetor unitário:
    v = u/|u|
   */

  double mod = VectorModule(*u);

  u->x = u->x/mod;
  u->y = u->y/mod;
  u->z = u->z/mod;

}

void Voxel2Vector(Voxel vxl, DVoxel *v) {
  /*
    Copia as coordenadas do voxel para o vetor
  */
  v->x = vxl.x;
  v->y = vxl.y;
  v->z = vxl.z;

}

DVoxel NormalVector(DVoxel u, DVoxel v) {
  /*
    Produto vetorial ou produto externo:
    u, v == (x, y, z)
    |  i  j  k |
    | ux uy uz | == (uy*vz - uz*vy)i + (uz*vx - ux*vz)j + (ux*vy - uy*vx)k
    | vx vy vz |
  */

  DVoxel n;

  n.x = u.y*v.z - u.z*v.y;
  n.y = u.z*v.x - u.x*v.z;
  n.z = u.x*v.y - u.y*v.x;

  return (n);
  
}

DVoxel Gauss2(double a[2][3], double x) {

  double m;
  DVoxel u;

  m = a[1][0]/a[0][0];
  a[1][0] = 0;
  a[1][1] = a[1][1] - m*a[0][1];
  a[1][2] = a[1][2] - m*a[0][2];

  u.x = x;
  u.y = (a[0][2] - a[0][1])/a[0][0];
  u.z = a[1][2]/a[1][1];

  return (u);
  
}


void Gauss3(double a[3][3], double lambda, DVoxel *u) {

  double m;
  int i, j, k, n;
  n = 3;

  for (k = 0; k < n - 1; k++) {
    for (i = k+1; i < n; i++) {
      m = a[i][k]/a[k][k];
      a[i][k] = 0;
      for (j = k+1; j < n; j++)
        a[i][j] = a[i][j] - m*a[k][j];
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
        printf("%7.2f\t", a[i][j]);
    }
    printf("\n");
  }

}


void ShowVector(double *u, int n) {
  
  int i;
  
  printf("Vetor = [");
  for (i = 0; i < n -1; i++)
    printf("%.5f, ",u[i]);
  printf("%.5f]\n", u[n-1]);
}

void ShowEquations(double a1, double b1, double c1, double r1,
                   double a2, double b2, double c2, double r2,
                   double a3, double b3, double c3, double r3) {

  printf("%7.2fx %7.2fy %7.2fz = %7.2f\n", a1, b1, c1, r1);
  printf("%7.2fx %7.2fy %7.2fz = %7.2f\n", a2, b2, c2, r2);
  printf("%7.2fx %7.2fy %7.2fz = %7.2f\n", a3, b3, c3, r3);

}


void ShowMatrixCramer(double a1, double b1, double c1,
                      double a2, double b2, double c2,
                      double a3, double b3, double c3) {

  printf("| %7.2f %7.2f %7.2f |\n", a1, b1, c1);
  printf("| %7.2f %7.2f %7.2f |\n", a2, b2, c2);
  printf("| %7.2f %7.2f %7.2f |\n", a3, b3, c3);

}

double Determinant(double a1, double b1, double c1,
                   double a2, double b2, double c2,
                   double a3, double b3, double c3) {

  return a1*b2*c3 + a2*b3*c1 + a3*b1*c2 - (a1*b3*c2 + a2*b1*c3 + a3*b2*c1);
  
}

void Cramer(double a1, double b1, double c1,
            double a2, double b2, double c2, 
            double a3, double b3, double c3) {
  
  double r1, r2, r3;
  double x, y, z, det;
  
  r1 = r2 = r3 = 0;
  
  det = Determinant(a1, b1, c1, a2, b2, c2, a3, b3, c3);
  
  if (det == 0) {
    ShowMatrixCramer(a1, b1, c1, a2, b2, c2, a3, b3, c3);
    printf("Determinante Zero! Impossível encontrar soluções!\n");
    return;
  }

  x = Determinant(r1, b1, c1, r2, b2, c2, r3, b3, c3)/det;
  y = Determinant(a1, r1, c1, a2, r2, c2, a3, r3, c3)/det;
  z = Determinant(a1, b1, r1, a2, b2, r2, a3, b3, r3)/det;

  return;
}


/* Método de resolução de equações cúbicas de Tartaglia */
void Tartaglia(double Sxx, double Syy, double Szz, 
               double Sxy, double Sxz, double Syz,
               double *r1, double *r2, double *r3) {

  double a, b, c, d;
  double r, t, p, q, u, v, u3, v3, im;
  double A, B, C, D, E, M, one_third;

  /* Equação resultante de det(S - x*I) == 0, */
  /* ax^3 + bx^2 + cx + d                     */
  /* x: escalar (Lambda)                      */
  /* I: Matriz Identidade                     */
  a = -1.0;
  b = Sxx + Syy + Szz;
  c = Sxy*Sxy + Sxz*Sxz + Syz*Syz - Sxx*Syy - Sxx*Szz - Syy*Szz;
  d = Sxx*Syy*Szz - Sxy*Sxy*Szz - Sxz*Sxz*Syy - Syz*Syz*Sxx + 2.0*Sxy*Sxz*Syz;

  A = b/a;
  B = c/a;
  C = d/a;

  one_third = 1.0/3.0;
  p = B - (A*A)/3.0;
  q = C - (A*B)/3.0 + 2.0*pow(A, 3)/27.0;

  D = pow(p, 3)/27.0 + q*q/4.0;

  if (D < 0) {
    M = sqrt((-1.0)*D);
    r = sqrt(q*q/4.0+M*M);
    t = acos((-1.0)*q/(2.0*r));
    *r1 = 2.0*pow(r, one_third)*cos(t/3.0)-A/3.0;
    *r2 = 2.0*pow(r, one_third)*cos((t+2*PI)/3.0)-A/3.0;
    *r3 = 2.0*pow(r, one_third)*cos((t+4*PI)/3.0)-A/3.0;
  }
  else {
    u3 = (-1.0)*q/2+sqrt(D);
    v3 = (-1.0)*q/2-sqrt(D);
    
    u = (u3 < 0) ? (-1)*pow((-1.0)*u3, one_third) : pow(u3, one_third);
    v = (v3 < 0) ? (-1)*pow((-1.0)*v3, one_third) : pow(v3, one_third);
    
    *r1 = u + v - A/3.0;

    E = pow(A + (*r1), 2.0) + 4*C/(*r1);
    if (E < 0) {
      im = sqrt(fabs(E))/2.0;
      *r2 = ((-1)*(A + (*r1)))/2.0;
      *r3 = ((-1)*(A + (*r1)))/2.0;
    }
    else {
      *r2 = ((-1)*(A + (*r1)) - sqrt(E))/2.0;
      *r3 = ((-1)*(A + (*r1)) + sqrt(E))/2.0;
    }
  }

  return;

}


void CovarianceMatrix(Scene *scn, Voxel center,
                      double *Sxx, double *Syy, double *Szz,
                      double *Sxy, double *Sxz, double *Syz) {

  int i, n, nv, xysize, xsize, x, y, z;

  xsize = scn->xsize;
  xysize = xsize*scn->ysize;
  n = xysize*scn->zsize;

  nv = *Sxx = *Syy = *Szz = *Sxy = *Sxz = *Syz = 0;
  for (i = 0; i < n; i++) {
    if (scn->data[i] != 0) {
      x = ((i % xysize) % xsize) - center.x;
      y = ((i % xysize) / xsize) - center.y;
      z = (i / xysize) - center.z;
      *Sxx += pow(x, 2);
      *Syy += pow(y, 2);
      *Szz += pow(z, 2);
      *Sxy += x*y;
      *Sxz += x*z;
      *Syz += y*z;
      nv++;
    }
  }

  *Sxx /= nv;
  *Syy /= nv;
  *Szz /= nv;
  *Sxy /= nv;
  *Sxz /= nv;
  *Syz /= nv;

  return;

}

/* Retorna o Voxel do centro de gravidade do objeto */
Voxel CenterOfGravity(Scene *scn) {

  Voxel center;
  int i, n, nv, xysize;

  center.x = center.y = center.z = 0;
  xysize = scn->xsize*scn->ysize;
  n = xysize*scn->zsize;
  nv = 0;
  for (i = 0; i < n; i++) {
    if (scn->data[i] != 0) {
      center.x += (i % xysize) % scn->xsize;
      center.y += (i % xysize) / scn->xsize;
      center.z += i / xysize;
      nv++;
    }
  }
  center.x /= nv;
  center.y /= nv;
  center.z /= nv;

  return (center);

}

Scene * RotateScene(Scene *scn, DVoxel normal, DVoxel center,
                    double shift, int debug)
{
  
  double costheta, sintheta;
  int i, j, k, q, p, max;
  Scene *res = NULL;
  DVoxel v, z, u, x;

  res = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  max = MaximumValue3(scn);

  normal.x *= scn->xsize; normal.y *= scn->xsize; normal.z *= scn->xsize;

  z.x = scn->xsize;
  z.y = 0;
  z.z = 0;

  costheta = GetCosine(normal, z);
  sintheta = GetSine(costheta);

  printf("Ângulo: %3.2f graus\n", 180*acos(costheta)/PI);

  x = NormalVector(normal, z);

  for (i = 0; i < scn->xsize; i++) {
    for (j = 0; j < scn->ysize; j++)
      for (k = 0; k < scn->zsize; k++) {
        u.x = i; u.y = j; u.z = k;
        v = RotateArbitrary(u, center, x, costheta, sintheta);
        p = i + scn->tby[j] + scn->tbz[k];
        q = ROUND(v.x) + scn->tby[ROUND(v.y)] + scn->tbz[ROUND(v.z)];
        if (ValidVoxel(res, v.x, v.y, v.z))
          res->data[p] = scn->data[q];
      }
  }

  if (debug) {
    WriteScene(res, "rotated.scn");
    printf("Cena rotacionada gravada em 'rotated.scn'\n");
  }
  
  return (res);
  
}


PlaneInfo CalculateSymmetry(Scene *scn, char *basename, char debug) {

  Voxel center;
  Scene *iscn=NULL, *plane=NULL;/* , *scn=NULL; */
  char filename[100];
  double Sxx, Syy, Szz, Sxy, Sxz, Syz, x;
  double r1, r2, r3, autovalue1, autovalue2;
  double a[2][3], s, shift, degrees, roll, yaw;
  DVoxel n, w, m, u, v;
  int i, slice_size=0, yaw_inf, yaw_sup, roll_inf, roll_sup, min;
  Slice *slice = NULL;
  PlaneInfo info;

/*   sprintf(filename, "%s.scn", basename); */
/*   scn = ReadScene(filename); */

  min = MIN(MIN(scn->dx, scn->dy), scn->dz);
  iscn = LinearInterp(scn, min, min, min);
/*   DestroyScene(&scn); */

  center = CenterOfGravity(iscn);

  /* Calcula a matriz de covariância da cena */
  CovarianceMatrix(iscn, center, &Sxx, &Syy, &Szz, &Sxy, &Sxz, &Syz);
  
  /* Encontra as soluções de uma equação de terceiro grau, */
  /* que são os autovalores da matriz de covariância       */
  Tartaglia(Sxx, Syy, Szz, Sxy, Sxz, Syz, &r1, &r2, &r3);
  
  /* Seleciona o maior e o menor autovalores */
  MinAndMax(r1, r2, r3, &autovalue1, &autovalue2);
  
  /* Assume-se que x possui valor 1 (sistema com 2 equações e 3 incógnitas) */
  x = 1.0;
  
  a[0][0] = Sxy;
  a[0][1] = Sxz;
  a[0][2] = autovalue1 - x*Sxx;
  a[1][0] = Syy - autovalue1;
  a[1][1] = Syz;
  a[1][2] = (-1)*x*Sxy;

  /* autovetor 1 */
  u = Gauss2(a, x);

  a[0][0] = Sxy;
  a[0][1] = Sxz;
  a[0][2] = autovalue2 - x*Sxx;
  a[1][0] = Syy - autovalue2;
  a[1][1] = Syz;
  a[1][2] = (-1)*x*Sxy;

  /* autovetor 2 */
  v = Gauss2(a, x);

  /* Converte em vetores unitários */
  VectorVersor(&u);
  VectorVersor(&v);

  /* Obtenção de 'n' pelo produto externo dos autovetores 'u' e 'v' */
  n = NormalVector(u, v);

  n.z /= n.x; n.y /= n.x; n.x /= n.x;

  /* Copia o voxel do centro de gravidade num vetor */
  Voxel2Vector(center, &w);

  shift = 2;
  degrees = 12;
  
  /* TESTE */
  /* fplane = GetPlane(iscn, n, w, 0); */
  /* WriteScene(fplane, "fplane.scn"); */

  /* De acordo com o sinal dos componentes 'y' e 'z' do vetor normal 'n',
     descobre-se em qual sentido deve-se girar para ajustar o plano
   */

  /* Y */
  if (n.y < 0) {
    yaw_inf = 0;
    yaw_sup = degrees;
  }
  else {
    yaw_inf = (-1)*degrees;
    yaw_sup = 0;
  }

  /* Z */
  if (n.z > 0) {
    roll_inf = 0;
    roll_sup = degrees;
  }
  else {
    roll_inf = (-1)*degrees;
    roll_sup = 0;
  }

  /*
    Desloca o eixo em 'shift' posições e
    rotaciona 'yaw' e 'roll' em 'degrees' graus para encontrar o melhor plano
  */
  slice_size = (shift+1)*(2*degrees+1)*(2*degrees+1);
  slice = (Slice *)calloc(slice_size, sizeof(Slice));
  
  i = 0;
  for (s = (-1)*(shift/2); s <= (shift/2); s++) {
    for (yaw = yaw_inf; yaw <= yaw_sup; yaw += 0.5) {
      for (roll = roll_inf; roll <= roll_sup; roll += 0.5) {
        RotateRoll(n, roll, &m);
        RotateYaw(m, yaw, &m);
        slice[i].intensity = GetPlaneIntensity(iscn, m, w, s);
        slice[i].roll = roll;
        slice[i].yaw = yaw;
        slice[i].shift = s;
        i++;
      }
    }
  }

  /* Ordena os cortes com QuickSort */
  qsort(slice, (size_t)slice_size, sizeof(Slice), (void *)CompareSlices);

  roll = slice[0].roll;
  yaw = slice[0].yaw;
  shift = slice[0].shift;

  RotateRoll(n, roll, &m);
  RotateYaw(m, yaw, &m);

  if (debug) {
    printf("DEBUG\n");
    plane = GetPlane(iscn, m, w, shift);
    sprintf(filename, "%s-plane.scn", basename);
    WriteScene(plane, filename);
    DestroyScene(&plane);
  }
  
  /*
    Armazena as coordenadas do vetor normal e do centro de gravidade do objeto
  */
  info.normal = m;
  info.center = w;
  info.shift = shift;

  free(slice);
  slice = NULL;
  DestroyScene(&iscn);

  return (info);

}
