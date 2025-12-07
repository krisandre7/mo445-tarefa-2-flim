#ifndef msp_location_h
#define msp_location_h 1

#include <iostream>

using namespace std;

namespace MSP{

//! Point in discrete 3D space
class Location {
 public:
  int X /** X coordinate */,Y /** Y coordinate */,Z /** Z coordinate */;

  //! Constructor
  Location() { X=Y=Z=0; }

  //! Constructor, sets (X,Y,Z)=(x,y,z)
  Location(int x,int y,int z) { X=x; Y=y; Z=z; }

  //! Copy constructor
  Location(  Location &src) { X=src.X; Y=src.Y; Z=src.Z; }

  //! Copy constructor
  Location(  Location *src) { X=src->X; Y=src->Y; Z=src->Z; }

  //! Translates this location by src, returns a reference to this
  //! location.
  Location & operator+=(  Location &src) {
    X+=src.X; Y+=src.Y; Z+=src.Z;
    return(*this);
  }

  //! Returns true if this location is closer to the origin than b.
  bool operator<(  Location &b)   {
    return(sqlen()<b.sqlen()); 
  }

  //! Returns true if this and b are the same location
  bool operator==(  Location &b)   {
    return(X==b.X && Y==b.Y && Z==b.Z); 
  }

  //! Returns true if this and b are different locations
  bool operator!=(  Location &b)   {
    return(X!=b.X || Y!=b.Y || Z!=b.Z); 
  }

  //! Assignment operator
  Location & operator=(  Location &b) { X=b.X; Y=b.Y; Z=b.Z; return(*this); }

  //! Sets (X,Y,Z) to (x,y,z), return a reference to this location.
  Location & set(int x,int y,int z) { X=x; Y=y; Z=z; return(*this); }
  
  //! Prints this location to stdout.
  void print() { cout << "(" << X << "," << Y << "," << Z << ") : " << sqlen() << "\n"; }

  //! Returns the square of the distance from the origin
  int sqlen()   { return( (X*X)+(Y*Y)+(Z*Z) ); }

  //! Returns the square of the distance between this and b
  int sqdist(  Location &b) {
    return( (X-b.X)*(X-b.X) + (Y-b.Y)*(Y-b.Y) + (Z-b.Z)*(Z-b.Z) );
  }
};

} //end MSP namespace

#endif

