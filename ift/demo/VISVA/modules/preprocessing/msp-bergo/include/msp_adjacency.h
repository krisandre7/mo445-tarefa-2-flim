#ifndef msp_adjacency_h
#define msp_adjacency_h 1

#include <vector>
#include "msp_location.h"

using namespace std;

namespace MSP{

//! Abstract discrete voxel adjacency
class Adjacency {
 public:
  //! Returns the number of elements in this adjacency
  int size() { return(data.size()); }

  //! Returns the index-th (0-based) neighbor displacement, as a \ref Location
  Location & operator[](int index) {
    return(data[index]);
  }

  //! Returns the i-th (0-based) neighbor of src, as a \ref Location
  Location & neighbor(  Location &src, int i) {
    static Location tmp;
    tmp = src;
    tmp += data[i];
    return(tmp);
  }

 protected:
  //! Vector of neighbors
  vector<Location> data;
};

//! Discrete spherical voxel adjacency
class SphericalAdjacency : public Adjacency {

 public:  
  //! Constructor
  SphericalAdjacency();

  //! Constructor, creates adjacency of given radius. If self is true,
  //! the center voxel is considered its own neighbor.
  SphericalAdjacency(float radius,bool self=false) {
    resize(radius,self);
  }

  //! Recreates this adjacency, with given radius. If self is true,
  //! the center voxel is considered its own neighbor.
  void resize(float radius, bool self=false) {
    int dx,dy,dz,r0,r2;

    data.clear();
    r0 = (int) radius;
    r2  = (int)(radius*radius);

    for(dz=-r0;dz<=r0;dz++)
      for(dy=-r0;dy<=r0;dy++)
	for(dx=-r0;dx<=r0;dx++) {
	  Location loc(dx,dy,dz);
	  if ((loc.sqlen() > 0  || self) && loc.sqlen() <= r2) {
	    Location *nloc = new Location(loc);
	    data.push_back(*nloc);
	  }
	}
    sort(data.begin(),data.end());
  }

};

//! Discrete circular adjacency on the XY plane
class DiscAdjacency : public Adjacency {

 public:
  //! Constructor
  DiscAdjacency();

  //! Constructor, creates adjacency of given radius. If self is true,
  //! the center voxel is considered its own neighbor.
  DiscAdjacency(float radius,bool self=false) {
    resize(radius,self);
  }

  //! Recreates this adjacency, with given radius. If self is true,
  //! the center voxel is considered its own neighbor.
  void resize(float radius, bool self=false) {
    int dx,dy,r0,r2;

    data.clear();
    r0 = (int) radius;
    r2  = (int)(radius*radius);

    for(dy=-r0;dy<=r0;dy++)
      for(dx=-r0;dx<=r0;dx++) {
	Location loc(dx,dy,0);
	if ((loc.sqlen() > 0  || self) && loc.sqlen() <= r2) {
	  Location *nloc = new Location(loc);
	  data.push_back(*nloc);
	}
      }
    sort(data.begin(),data.end());
  }

};

} //end MSP namespace

#endif
