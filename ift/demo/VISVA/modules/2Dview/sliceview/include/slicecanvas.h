
#ifndef _SLICECANVAS_H_
#define _SLICECANVAS_H_

#include "startnewmodule.h"

namespace SliceView{

    typedef enum {HIGHLIGHT_FILL=0,
        HIGHLIGHT_BORDER,
        HIGHLIGHT_INSIDE,
        HIGHLIGHT_OFF} HighlightType;

    typedef enum {DATA_ORIG=0,
        DATA_ARCW,
        DATA_GRAD,
        DATA_OBJMAP,
        DATA_OBJ,
        DATA_CUSTOM} DataType;

    typedef enum {MARKER_ON=0,
        MARKER_OFF} MarkerType;

    class SliceCanvas : public Canvas {
        public:
        DataType data;
        Scene *custom_scn;

        SliceCanvas(wxWindow *parent);
        virtual ~SliceCanvas();
        void Zoomin();
        void Zoomout();
        void SetZoomLevel(float zoom);
        void SetSliceAxis(char axis);
        char GetSliceAxis();
        void Draw();
        void DrawScnGradient();
        void SetRefreshHandler(RefreshHandler *handler);
        int  Slice2voxel(int x, int y, Voxel *out);
        int  Canvas2Address(int x, int y);
        int  Canvas2voxel(int x, int y, Voxel *out);
        void ChangeDrawMarker();
        void SetInteractionHandler(InteractionHandler *handler);
        void SetCustomData(Scene *scn);
        void Set2DViewOptions(HighlightType highlight,
                              DataType data,
                              MarkerType marker);
        void OnMouseEvent(wxMouseEvent& event);
        protected:
        InteractionHandler *defaulthandler;
        RefreshHandler *drawhandler; /* Callback */
        char axis;
        bool drawmarker;
        HighlightType highlight;
        MarkerType marker;
        Voxel GetCutVoxel();
        int  xgray(int value);
        void FindImageSize(int *w, int *h);
        void RotateNTimes(int *rj, int *ri,
                          int j, int i,
                          int w, int h,
                          int ntimes);
        void RotateBackNTimes(int *j, int *i,
                              int rj, int ri,
                              int rw, int rh,
                              int ntimes);
        void DrawOverlay();
    };
} //end SliceView namespace

#include "sliceview.h"

#endif

