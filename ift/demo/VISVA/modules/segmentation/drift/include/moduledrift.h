
#ifndef _MODULEDRIFT_H_
#define _MODULEDRIFT_H_

#include "startnewmodule.h"
//extern "C" {
    //#include "ift.h"
//}

#define MAX_OBJS 100000

namespace DRIFT{

  class ModuleDRIFT : public SegmentationModule {
  public:

    ModuleDRIFT();
    ~ModuleDRIFT();
    void Start();
    bool Stop();
    void Finish();
    void Run();
    iftSet* ProcessSeeds(iftLabeledSet **LS);
    iftFImage ** pvtCreateObjectEDTFromGT(int number_of_objects, iftImage *gt_image);
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

    int id_spin;
    int id_check;
    wxSpinCtrl    *spin;
    wxCheckBox    *checksmooth;

    //DIFT data:
    Scene *pred;
    bia::Scene16::Scene16 *cost;
    bia::Scene16::Scene16 *Wf;


    /* New IFT variables */
    int largest_id, execution;
    iftImage *iftorig;
    iftImage *gradient;
    iftImage *seedimage;
    iftImage *gui_markers;
    iftSmoothBorder *smooth;
    iftImageForest *forest;

  protected:
    void InitNull();
    void Run_DIFT();
    void SetWf(Scene *Wl, Scene *bin);
    void ComputeGradientImageBasins();
    void AllocData();
    void CreateDiffusionFilter();
    void FreeData();
    void PrintSeedReport();

    bool active;
    //wxPanel *optPanel;
    BaseDialog *optDialog;
  };

} //end DRIFT namespace



#endif

