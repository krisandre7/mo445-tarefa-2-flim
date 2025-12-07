#ifndef msp_volumedomain_h
#define msp_volumedomain_h 1

#include <math.h>

namespace MSP{

//! A 3D discrete domain
class VolumeDomain {

 public:

  int W,H,D,N,WxH;

  //! Constructor
  VolumeDomain() {
    tby = NULL;
    tbz = NULL;
    W = H = D = WxH = N = 0;
  }
  
  //! Constructor, creates a domain with given width, height and depth
  VolumeDomain(int w,int h,int d) {
    tby = NULL;
    tbz = NULL;
    resize(w,h,d);
  }

  //! Destructor
  virtual ~VolumeDomain() {
    if (tby) delete tby;
    if (tbz) delete tbz;
  }

  //! Resizes this domain
  virtual void resize(int w,int h,int d) {
    int i;

    W = w;
    H = h;
    D = d;
    N = W*H*D;
    WxH = W*H;
    if (tby!=NULL) delete tby;
    if (tbz!=NULL) delete tbz;
    if (N==0) {
      tby = NULL;
      tbz = NULL;
      return;
    }
    tby = new int[H];
    tbz = new int[D];
    for(i=0;i<H;i++) tby[i] = W*i;
    for(i=0;i<D;i++) tbz[i] = WxH*i;
  }

  //! Returns the linear address of voxel (x,y,z)
  int  address(int x,int y,int z) {
    return(x+tby[y]+tbz[z]);
  }

  //! Returns the linear address of \ref Location loc
  int  address(  Location &loc) {
    return(loc.X+tby[loc.Y]+tbz[loc.Z]);
  }

  //! Returns true if voxel (x,y,z) is within this domain
  bool valid(int x,int y,int z) {
    return((x>=0)&&(y>=0)&&(z>=0)&&(x<W)&&(y<H)&&(z<D));
  }

  //! Returns true if voxel (x,y,z) is within this domain
  bool valid(float x,float y,float z) {
    return((x>=0)&&(y>=0)&&(z>=0)&&(x<W)&&(y<H)&&(z<D));
  }

  //! Returns true if \ref Location loc is within this domain
  bool valid(  Location &loc) {
    return(valid(loc.X,loc.Y,loc.Z));
  }
  
  //! Returns the X component of the linear address a
  int  xOf(int a) { return( (a%WxH) % W ); }

  //! Returns the Y component of the linear address a
  int  yOf(int a) { return( (a%WxH) / W ); }

  //! Returns the Z component of the linear address a
  int  zOf(int a) { return( a / WxH ); }

  //! Returns the length of the diagonal segment (0,0,0)-(W,H,D)
  int diagonalLength() { return((int) (sqrt(W*W+H*H+D*D))); }

 private:
  int *tby, *tbz;
};


} //end MSP namespace

#endif
