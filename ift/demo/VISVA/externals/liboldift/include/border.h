#ifndef _BORDER_
#define _BORDER_


typedef struct _mybordernode {
	int next; // index of next voxel in the border voxel list
	int prev; // index of previous voxel in the border voxel list
} BorderNode;

typedef struct _myborder {
  BorderNode *voxels; // doubly linked list of border voxels
  int         ncur;   // current number of voxels in the border
  int         nmax;   // maximum number of voxels in the border
  int         first;  // first voxel of the border voxel list
  int         last;   // last voxel of the border voxel list
} Border;

Border *CreateBorder(int n);
void    DestroyBorder(Border **border);
void    InsertBorder(Border *border, int index);
void    RemoveBorder(Border *border, int index);
int     IsBorder(Border *border, int index);

#endif
