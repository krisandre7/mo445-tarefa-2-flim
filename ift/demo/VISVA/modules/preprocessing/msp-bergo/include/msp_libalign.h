
#ifndef msp_libalign_h
#define msp_libalign_h 1

#include "msp_volume.h"

namespace MSP{

/*
  align: biosignals algorithm

  input parameters:
     input:       input volume (sagittal)
     out:         filename to save volume with drawn plane
     debugprefix: if not null, intermediary volumes are
                  saved with this prefix 
     orientation: input orientation (use UnknownOrientation to guess,
                                     currently broken)
     verbose:     output optimization iterations to stdout
     
     scale:       scale factor before segmentation (< 1 for reduced resolution)

  output parameters:
     ntrans: number of fitting iterations required
     trans:  the resulting plane transform
     planez: (Z=planez) x (trans) is the symmetry plane
     tseg:   time elapsed segmenting the brain (secs)
     tali:   time elapsed fitting the plane (secs)
*/

void align(Volume<int> *input, 
	   const char *out, 
	   const char *debugprefix,
	   VolumeOrientation orientation,
	   int &ntrans,
	   T4 &trans, 
           int &planez,
	   double &tseg,
	   double &tali,
           bool verbose,
	   float scale);

/* 
   align2: EDT-based alignment.
           Parameter semantic is the same of align
*/
void align2(Volume<int> *input,
	    const char *out, 
	    const char *debugprefix,
	    VolumeOrientation orientation,
	    int &ntrans,
	    T4 &trans, 
	    int &planez,
	    double &tseg,
	    double &tali,
	    bool verbose,
	    float scale);

/* 
   align3: Volkau'06's method
           Parameter semantic is the same of align

   maxdelta: 3 = volkau original
*/
void align3(Volume<int> *input,
	    const char *out, 
	    const char *debugprefix,
	    VolumeOrientation orientation,
	    int &ntrans,
	    T4 &trans, 
	    int &planez,
	    double &tseg,
	    double &tali,
	    bool verbose,
	    int maxdelta);

Volume<int> * toSagittal(Volume<int> *src, 
			 VolumeOrientation orientation);

char *namestrip(const char *src);

T4 & randomTrans(int cx, int cy, int cz);


} //end MSP namespace

#endif
