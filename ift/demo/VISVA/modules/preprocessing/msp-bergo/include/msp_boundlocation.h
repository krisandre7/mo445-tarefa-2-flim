#ifndef msp_boundlocation_h
#define msp_boundlocation_h 1

#include <iostream>
#include "msp_location.h"
#include "msp_volumedomain.h"

using namespace std;

namespace MSP{

//! Point in bound discrete 3D space
class BoundLocation : public Location {
 public:
  //! Constructor
  BoundLocation() : Location() { bind(0,0,0); }

  //! Constructor
  BoundLocation(VolumeDomain &vd) : Location() {
    bind(vd);
  }
    
  //! Constructor
  BoundLocation(int w,int h,int d) : Location() {
    bind(w,h,d);
  }

  //! Copy constructor
  BoundLocation(BoundLocation &b) {
    (*this) = b;
  }

  virtual ~BoundLocation() { }

  //! Binds this location to a \ref VolumeDomain
  void bind(VolumeDomain &vd) {
    W = vd.W; H = vd.H; D = vd.D;
    postBind();
  }

  //! Binds this location to a (w,h,d) domain
  void bind(int w,int h,int d) {
    W = w; H = h; D = d;
    postBind();
  }

  //! Advances to the next location within the bound domain
  BoundLocation & operator++(int a) {
    ++X; 
    if (X==W) { X=0; ++Y; }
    if (Y==H) { Y=0; Z++; } 
    if (Z>=D) { X=0; Y=0; Z=D; } 
    return(*this);
  }

  //! Assignment operator
  BoundLocation & operator=(  BoundLocation &b) {
    X=b.X; Y=b.Y; Z=b.Z; bind(b.W,b.H,b.D); return(*this); }

  //! Assignment operator
  BoundLocation & operator=(  Location &b) {
    X=b.X; Y=b.Y; Z=b.Z; return(*this); }

  //! Returns a reference to the first location in the bound domain
  Location & begin() { return pb; }

  //! Returns a reference to the location after the last one in the bound domain
  Location & end() { return pe; }

  //! Prints this location to stdout.
  virtual void print() { cout << "(" << X << "," << Y << "," << Z << "), (" << W << "," << H << "," << D << ")" << endl; }


 private:
  int W,H,D;
  Location pb, pe;

  void postBind() {
    pb.set(0,0,0);
    pe.set(0,0,D);
  }

};


} //end MSP namespace

#endif
