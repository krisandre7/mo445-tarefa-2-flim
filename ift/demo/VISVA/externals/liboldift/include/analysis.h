#ifndef _ANALYSIS_H_
#define _ANALYSIS_H_

#include "image.h"
#include "annimg.h"
#include "curve.h"
#include "polynom.h"
#include "gqueue.h"

Image   *Perimeter(Image *bin);
Image   *Area(Image *bin);
Image   *GeomCenter(Image *bin);
Image   *CompMSSkel(AnnImg *aimg);
Image   *Skeleton(Image *msskel, float perc);
Image   *CompSKIZ(AnnImg *aimg);
Curve3D *CompSaliences(AnnImg *aimg, int maxcost);
Curve3D *RemSaliencesByAngle(Curve3D *curve,int radius, int angle);
Image   *PaintSaliences(Image *bin, Curve3D *curve);

/* -------------------- IFT-based Operators ------------------------ */

void     iftFastDilation(AnnImg *aimg, AdjRel *A); /* by Cuisenaire */
void     iftDilation(AnnImg *aimg, AdjRel *A); /* by Dial */
void     iftDilationHeap(AnnImg *aimg, AdjRel *A); /* by Heap */
int      iftGeoDilation(AnnImg *aimg); /* using Chamfer 17x24. Source set(s) must be 1 and destination set(s) must be 2. */
//Image   *iftDistTrans(AnnImg *aimg, AdjRel *A, char side); 
Image   *DistTrans(Image *bin, AdjRel *A, char side); 
Image   *SignedDistTrans(Image *bin, AdjRel *A, char side);
int      iftGeoDist(AnnImg *aimg); 
int     *iftGeoPath(AnnImg *aimg);
Polynom *MSFractal(Image *bin,int maxdist,int degree,double lower,double higher,int reg,double from,double to);
Image   *MSSkel(Image *bin, char side); 
Image   *SKIZ(Image *bin, char side);
Curve3D *Saliences(Image *bin, int maxdist); /* If maxdist is
                                                   longer than the
                                                   shortest distance
                                                   between object and
                                                   image frame, it
                                                   requires MBB and
                                                   AddFrame before */

Curve3D *SkelSaliences(Image *skel, int maxdist, int angle); 
Image   *PaintSkelSaliences(Image *skel, int angle);
Curve3D *ContSaliences(Image *bin,int maxdist,int threshold,int angle);
Curve3D *SkelCont(Image *bin, int maxdist, int threshold, int angle, char side);
Image   *LabelSkel(Image *skel, Curve3D *curve, char option);

Image   *ObjectBorder(Image *bin);

/**
 * Calcula a maior e menor distância geodésica entre pontos
 * opostos do contorno.
 * A função assume que há somente um objeto na imagem.
 **/
void GeoDistMaxMinBin(Image *bin, int *max, int *min);

// Compute contour salience points with salience values. The parameter
// thres must be defined between 0 and 180.

Curve3D *ContourSaliencePoints(Image *bin, float thres);


#endif  







