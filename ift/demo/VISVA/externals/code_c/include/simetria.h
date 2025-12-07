#ifndef _SIMETRIA_H_
#define _SIMETRIA_H_

#include "oldift.h"

typedef struct _slice {
  double roll;
  double yaw;
  double shift;
  int intensity;
} Slice;

typedef struct _dvoxel {
  double x;
  double y;
  double z;
} DVoxel;

typedef struct _planeinfo {
  DVoxel normal;
  DVoxel center;
  double shift;
} PlaneInfo;


double GetCosine(DVoxel u, DVoxel v);

double GetSine(double cosine);

double GetAngle(double cosine);

DVoxel RotateArbitrary(DVoxel p, DVoxel c, DVoxel n,
                       double costheta, double sintheta);

void Normalise(DVoxel *v);

Scene * RotateScene(Scene *scn, DVoxel normal, DVoxel center,
                    double shift, int debug);

void VectorReflection(DVoxel p1, DVoxel center, DVoxel n, DVoxel *p);

void WritePlaneInfo(char *filename, DVoxel n, DVoxel center, double x_shift);

int ReadPlaneInfo(char *filename, DVoxel *n, DVoxel *center, double *x_shift);

double GetCosine(DVoxel u, DVoxel v);

int CompareSlices(Slice *slc1, Slice *slc2);

void RotatePitch(DVoxel n, double degree, DVoxel *m);

void RotateRoll(DVoxel n, double degree, DVoxel *m);

void RotateYaw(DVoxel n, double degree, DVoxel *m);

void MinAndMax(double n1, double n2, double n3, double *max1, double *max2);

void MinTwo(double n1, double n2, double n3, double * max1, double * max2);

void MaxTwo(double n1, double n2, double n3, double * max1, double * max2);

void Median(double n1, double n2, double n3, double *m);

double InternalProduct(DVoxel u, DVoxel v);

Scene * GetPlane(Scene *scn, DVoxel n, DVoxel p, int shift);

int GetPlaneIntensity(Scene *scn, DVoxel n, DVoxel c, int shift);

double VectorModule(DVoxel v);

void VectorVersor(DVoxel *u);

void Voxel2Vector(Voxel vxl, DVoxel *v);

DVoxel NormalVector(DVoxel u, DVoxel v);

DVoxel Gauss2(double a[2][3], double x);

void Gauss3(double a[3][3], double lambda, DVoxel *u);

void ShowVector(double *u, int n);

void ShowEquations(double a1, double b1, double c1, double r1,
                   double a2, double b2, double c2, double r2,
                   double a3, double b3, double c3, double r3);

void ShowMatrixCramer(double a1, double b1, double c1,
                      double a2, double b2, double c2,
                      double a3, double b3, double c3);

double Determinant(double a1, double b1, double c1,
                   double a2, double b2, double c2,
                   double a3, double b3, double c3);

void Cramer(double a1, double b1, double c1,
            double a2, double b2, double c2, 
            double a3, double b3, double c3);

void Tartaglia(double Sxx, double Syy, double Szz, 
               double Sxy, double Sxz, double Syz,
               double *r1, double *r2, double *r3);

void CovarianceMatrix(Scene *scn, Voxel center,
                      double *Sxx, double *Syy, double *Szz,
                      double *Sxy, double *Sxz, double *Syz);

Voxel CenterOfGravity(Scene *scn);

PlaneInfo CalculateSymmetry(Scene *scn, char * basename, char debug);

#endif
