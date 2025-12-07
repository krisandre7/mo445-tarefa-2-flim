#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

typedef struct _vector{
  float x;
  float y;
  float z;
} Vector, Point, Vertex;

typedef struct _edge {
  int vert[2];
} Edge;

typedef struct _myquad {
  int vert[4];
} Quad;


float    ScalarProd(Vector v1, Vector v2);
Vector   VectorProd(Vector v1, Vector v2);
Vector   VectorRotate(Vector v1, float R[4][4]);
void     VectorNormalize(Vector *v);
float    VectorMagnitude(Vector *v);
#endif
