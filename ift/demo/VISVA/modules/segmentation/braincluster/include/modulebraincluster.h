
#ifndef _MODULEBRAINCLUSTER_H_
#define _MODULEBRAINCLUSTER_H_

#include "startnewmodule.h"
#include "braincluster_dialog.h"


namespace BrainCluster
{

  //  class ModuleMSP : public RegistrationModule
  class ModuleBrainCluster : public SegmentationModule
  {
    
  public:
    ModuleBrainCluster();
    ~ModuleBrainCluster();
    
    void Start();
    bool Stop();
    void AddMaskToSegmObj(Scene *mask, char *obj_name, int obj_color);

  protected:
    wxDialog *dialog;
    
  };



} //end namespace



#endif

