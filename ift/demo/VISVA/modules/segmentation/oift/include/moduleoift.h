
#ifndef _MODULEOIFT_H_
#define _MODULEOIFT_H_

#include "iftIGraph.h"
#include "startnewmodule.h"
//extern "C" {
//#include "ift.h"
//}

#define MAX_OBJS 100000

namespace OIFT{

    class ModuleOIFT : public SegmentationModule {
    public:
        enum OIFTType {
            Watershed,
            OIFTResuming,
            ISFResuming
        };


        ModuleOIFT();
        ~ModuleOIFT();
        void Start();
        bool Stop();
        void Finish();
        void Run();

        iftFImage ** pvtCreateObjectEDTFromGT(int number_of_objects, iftImage *gt_image);
        void Relax();
        void Reset();
        AdjRel   *GetBrush();
        wxCursor *GetBrushCursor(int zoom);
        void      NextBrush();
        void      PrevBrush();
        void      DeleteObj(int obj);

        int GetCodeID(int cod);
        int GetCodeLabel(int cod);
        int GetCodeValue(int id, int lb);

        char obj_name[MAX_OBJS][128];
        int  obj_color[MAX_OBJS];
        int  obj_visibility[MAX_OBJS];
        int  obj_sel;
        int  nobjs;
        int  markerID;


        /* New IFT variables */
        int highest_id, execution;

        OIFTType del_alg_type;

        // ISF variables
        double alpha;
        double beta;
        double gamma;
        int niters;

        // OIFT watershed variables
        double ori_watershed_gamma;
        int nseeds_per_label_per_iteration;
        double min_safe_distance_to_border;
        double max_marker_width;
        double max_marker_length;
        double stopping_threshold;
        double sec_stopping_threshold;


protected:

        iftImage *iftorig;
        iftImage *gradient;
        iftImage *presegmentation;

        iftImageForest *fst;
        iftIGraph *igraph;
        iftDHeap *Q;

        void InitNull();

        void AllocBasicData();
        void AllocPriorityQueue();
        void ComputeGradientImageBasins(iftAdjRel *A);
        iftSet* ProcessSeeds(iftLabeledSet **LS, int last_markerID);
        void DisplayParamPanel();
        void DisplayPanel();
        void CreateDiffusionFilter();
        void FreeData();
        void PrintSeedReport();

        bool active;


        //wxPanel *optPanel;
        BaseDialog *optDialog;
};

} //end OIFT namespace



#endif

