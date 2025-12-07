
#ifndef _BIA_SEGMENTATION3_H_
#define _BIA_SEGMENTATION3_H_

#include "bia_common.h"
#include "bia_scene.h"
#include "bia_adjrel3.h"


namespace bia{

  namespace Scene8{

    Scene8 *Threshold(Scene8 *scn, int lower, int higher);
    Scene8 *GetBoundaries(Scene8 *scn, AdjRel3::AdjRel3 *A);

  } //end Scene8 namespace



  namespace Scene16{

    Scene8::Scene8 *Threshold(Scene16 *scn, int lower, int higher);

  } //end Scene16 namespace



  namespace Scene32{

    Scene8::Scene8 *Threshold(Scene32 *scn, int lower, int higher);

  } //end Scene32 namespace


  namespace Scene{

    Scene8::Scene8 *Threshold(Scene *scn, int lower, int higher);

  } //end Scene namespace


} //end bia namespace


#endif

