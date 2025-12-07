
#ifndef _MODULEINTERACTIVE_H_
#define _MODULEINTERACTIVE_H_

#include "startnewmodule.h"

#define MAX_OBJS 100000

namespace Interactive{

  class ModuleInteractive : public SegmentationModule{
  public:

    ModuleInteractive();
    ~ModuleInteractive();
    void Start();
    bool Stop();
    void Finish();
    void Run();
    void Relax();
    void Reset();
    AdjRel   *GetBrush();
    wxCursor *GetBrushCursor(int zoom);
    void      NextBrush();
    void      PrevBrush();
    void      DeleteObj(int obj);
    void      MarkForRemovalIsolatedSeeds();

    int GetCodeID(int cod);
    int GetCodeLabel(int cod);
    int GetCodeValue(int id, int lb);

    char obj_name[MAX_OBJS][128];
    int  obj_color[MAX_OBJS];
    int  obj_visibility[MAX_OBJS];
    int  obj_sel;
    int  nobjs;
    int  markerID;

    //DIFT data:
    Scene *pred;
    bia::Scene16::Scene16 *cost;
    bia::Scene16::Scene16 *Wf;

  protected:
    void Run_DIFT();
    void SetWf(Scene *Wl, Scene *bin);
    void ComputeArcWeight();
    void AllocData();
    void FreeData();
    void PrintSeedReport();

    bool active;
    //wxPanel *optPanel;
    BaseDialog *optDialog;
  };

} //end Interactive namespace

#endif

