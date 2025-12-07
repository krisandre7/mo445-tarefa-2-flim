
#ifndef msp_align_h
#define msp_align_h 1

#include "msp_volume.h"
#include "msp_libalign.h"

namespace MSP{

void usage();

void imgAlign(const char *in, 
	      const char *out, 
	      T4 &itrans, 
	      VolumeOrientation orientation,
	      int iw,int ih,int id, 
	      float idx, float idy, float idz,
	      float vs, bool verbose, bool pad);

int msp_align(char *command);


} //end MSP namespace

#endif

