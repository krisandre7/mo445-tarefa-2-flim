
#ifndef _MODULEENHANCEMENT_H_
#define _MODULEENHANCEMENT_H_

#include "startnewmodule.h"


namespace Enhancement{

  class ModuleEnhancement : public PreProcModule{
  public:

    ModuleEnhancement();
    ~ModuleEnhancement();
    void Start();
    bool Stop();
    void Finish();
    void Run();
    void Reset();
    AdjRel   *GetBrush();
    wxCursor *GetBrushCursor(int zoom);
    void      NextBrush();
    void      PrevBrush();

    int GetCodeID(int cod);
    int GetCodeLabel(int cod);
    int GetCodeValue(int id, int lb);

    int  markerID;

  protected:
    void AllocData();
    void FreeData();
    
    bool active;
    BaseDialog *optDialog;
    Scene *gradi;
  };

} //end Enhancement namespace

#endif

