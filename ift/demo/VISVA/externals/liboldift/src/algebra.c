#include "algebra.h"
#include "common.h"

char PAxis(Vector viewer)
{
  Vector viewer_aux;

  viewer_aux.x = fabs(viewer.x);
  viewer_aux.y = fabs(viewer.y);
  viewer_aux.z = fabs(viewer.z);
  
  if ((viewer_aux.x >= viewer_aux.y) && 
      (viewer_aux.x >= viewer_aux.z))
    return('x');
  else
    if ((viewer_aux.y >= viewer_aux.x) && 
	(viewer_aux.y >= viewer_aux.z))
      return('y');
    else
      return('z');
}

char Octant(Vector viewer)
{
  Vector oct[8];
  int i;
  char o=-1;
  float p,maxp=-2.0;

  oct[0].x = -1.0; oct[0].y = -1.0; oct[0].z = -1.0; 
  oct[1].x =  1.0; oct[1].y = -1.0; oct[1].z = -1.0; 
  oct[2].x = -1.0; oct[2].y =  1.0; oct[2].z = -1.0; 
  oct[3].x =  1.0; oct[3].y =  1.0; oct[3].z = -1.0; 
  oct[4].x = -1.0; oct[4].y = -1.0; oct[4].z =  1.0; 
  oct[5].x =  1.0; oct[5].y = -1.0; oct[5].z =  1.0; 
  oct[6].x = -1.0; oct[6].y =  1.0; oct[6].z =  1.0; 
  oct[7].x =  1.0; oct[7].y =  1.0; oct[7].z =  1.0; 

  for (i=0; i < 8; i++) {
    p = oct[i].x*viewer.x + oct[i].y*viewer.y + oct[i].z*viewer.z;
    if (p > maxp){
      maxp = p;
      o = i+1;
    }
  }
  return(o);
}


void MultMatrices(float M1[4][4],float M2[4][4], float M3[4][4])
{
  int i,j,k;


  for (k=0; k < 4; k++)
    for (j=0; j < 4; j++)
      M3[k][j] = 0.0;

  for (k=0; k < 4; k++)
    for (j=0; j < 4; j++)
      for (i=0; i < 4; i++)
				M3[k][j]+=M1[k][i]*M2[i][j];

}

void TransMatrix(float M1[4][4],float M2[4][4])
{
  int i,j;

  for (j=0; j < 4; j++)
    for (i=0; i < 4; i++)
      M2[i][j]=M1[j][i];
}

Point RotatePoint(float M[4][4], Point p)
{
  Point np;

  np.x = M[0][0]*p.x + M[0][1]*p.y + M[0][2]*p.z;
  np.y = M[1][0]*p.x + M[1][1]*p.y + M[1][2]*p.z;
  np.z = M[2][0]*p.x + M[2][1]*p.y + M[2][2]*p.z;

  return(np);
}

Point TransformPoint(float M[4][4], Point p)
{
  Point np;

  np.x = M[0][0]*p.x + M[0][1]*p.y + M[0][2]*p.z + M[0][3];
  np.y = M[1][0]*p.x + M[1][1]*p.y + M[1][2]*p.z + M[1][3];
  np.z = M[2][0]*p.x + M[2][1]*p.y + M[2][2]*p.z + M[2][3];

  return(np);
}
