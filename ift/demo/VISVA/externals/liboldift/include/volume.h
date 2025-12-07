#ifndef _VOLUME_H_
#define _VOLUME_H_

#include "image.h"

typedef struct _volume {
  int *val;
  int ncols,nrows,nimgs;
  int *tbrow;
  int *tbimg;
} Volume;

Volume  *CreateVolume(int ncols,int nrows,int nimgs);
void     DestroyVolume(Volume **vol);
Volume  *Lift(Image *img);
Image   *Flat(Volume *vol);
Volume  *CopyVolume(Volume *vol);
void     SetVolume(Volume *vol, int value);
int      VolumeMax(Volume *vol);

/*
#define SetVoxel(vol,n,value) (vol)->val[(n)]=value
#define GetVoxel(vol,n)       ((vol)->val[(n)])

#define VoxelAddress(vol,x,y,z) ((x)+(vol)->tbrow[(y)]+(vol)->tbimg[(z)])
#define VoxelX(vol,n) (((n) % (((vol)->ncols)*((vol)->nrows))) % (vol)->ncols)
#define VoxelY(vol,n) (((n) % (((vol)->ncols)*((vol)->nrows))) / (vol)->ncols)
#define VoxelZ(vol,n) ((n) / (((vol)->ncols)*((vol)->nrows)))
*/
#define VolumeLen(vol) ((vol)->ncols * (vol)->nrows * (vol)->nimgs)


#endif








