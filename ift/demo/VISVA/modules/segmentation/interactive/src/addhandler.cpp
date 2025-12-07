
#include "addhandler.h"

namespace Interactive{

    AddHandler :: AddHandler(char axis,
                             ModuleInteractive *mod){
        this->axis = axis;
        this->mod = mod;
    }

    AddHandler :: ~AddHandler(){}

    void AddHandler :: OnEnterWindow(){
        wxCursor *cursor;
        float zoom = APP->GetZoomLevel();

        cursor = mod->GetBrushCursor(ROUND(zoom));
        APP->Set2DViewCursor(cursor, axis);
    }

    void AddHandler :: OnLeaveWindow(){
        APP->Set2DViewCursor(wxSTANDARD_CURSOR, axis);
    }

    void AddHandler :: OnLeftClick(int p) {
        AdjRel *A;
        int o, cod;

        o = mod->obj_sel;
        if (o < 0) return;

        mod->markerID++;
        cod = mod->GetCodeValue(mod->markerID, o+1);
        APP->SetLabelColour(cod, mod->obj_color[o]);
        A = mod->GetBrush();
        APP->AddSeedsInBrush(p, o+1, mod->markerID,
                             A, axis);
        APP->DrawBrush(p, cod, A, axis);
        //Reset the cost to force seed execution:
        APP->DrawBrushCustom(p, INT_MAX, A, axis, mod->cost);

        DestroyAdjRel(&A);
        APP->Refresh2DCanvas();
    }

    void AddHandler :: OnRightClick(int p){
        AdjRel *A;
        int cod;

        mod->markerID++;
        cod = mod->GetCodeValue(mod->markerID, 0);
        APP->SetLabelColour(cod, NIL);
        A = mod->GetBrush();
        APP->AddSeedsInBrush(p, 0, mod->markerID,
                             A, axis);
        APP->DrawBrush(p, cod, A, axis);
        //Reset the cost to force seed execution:
        APP->DrawBrushCustom(p, INT_MAX, A, axis, mod->cost);

        DestroyAdjRel(&A);
        APP->Refresh2DCanvas();
    }

    void AddHandler :: OnMiddleClick(int p){
        mod->Run();
    }


    void AddHandler :: OnMouseWheel(int p, int rotation, int delta){
        if(rotation>0) mod->NextBrush();
        else           mod->PrevBrush();

        wxCursor *cursor;
        float zoom = APP->GetZoomLevel();

        cursor = mod->GetBrushCursor(ROUND(zoom));
        APP->Set2DViewCursor(cursor, axis);
    }


    void AddHandler :: OnLeftDrag(int p, int q){
        AdjRel *A;
        int o,cod;

        o = mod->obj_sel;
        if(o<0) return;
        cod = mod->GetCodeValue(mod->markerID, o+1);
        A = mod->GetBrush();
        APP->AddSeedsInBrushTrace(p,q,o+1,mod->markerID,
                                  A,axis);
        APP->DrawBrushTrace(p,q,cod,A,axis);
        //Reset the cost to force seed execution:
        APP->DrawBrushTraceCustom(p,q,INT_MAX,A,axis,mod->cost);

        DestroyAdjRel(&A);
        APP->Refresh2DCanvas();
    }

    void AddHandler :: OnRightDrag(int p, int q){
        AdjRel *A;
        int cod;

        cod = mod->GetCodeValue(mod->markerID, 0);
        A = mod->GetBrush();
        APP->AddSeedsInBrushTrace(p,q,0,mod->markerID,
                                  A,axis);
        APP->DrawBrushTrace(p,q,cod,A,axis);
        //Reset the cost to force seed execution:
        APP->DrawBrushTraceCustom(p,q,INT_MAX,A,axis,mod->cost);

        DestroyAdjRel(&A);
        APP->Refresh2DCanvas();
    }

    void AddHandler :: OnMouseMotion(int p){
        wxCursor *cursor=NULL;
        static float zoom = -1.0;

        if(zoom!=APP->GetZoomLevel()){
            zoom = APP->GetZoomLevel();
            cursor = mod->GetBrushCursor(ROUND(zoom));
            APP->Set2DViewCursor(cursor, axis);
        }

        APP->Window->SetStatusText(_T("Mouse Left: Add Object Marker, Mouse Center: Run, Mouse Right: Add Background Marker"));
    }

} //end Interactive namespace
