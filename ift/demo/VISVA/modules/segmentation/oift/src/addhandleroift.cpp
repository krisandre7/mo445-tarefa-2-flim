
#include "addhandleroift.h"

namespace OIFT
{

AddHandlerOIFT :: AddHandlerOIFT(char axis,
                                   ModuleOIFT *mod)
{
    this->axis = axis;
    this->mod = mod;
}

AddHandlerOIFT :: ~AddHandlerOIFT() {}

void AddHandlerOIFT :: OnEnterWindow()
{
    wxCursor *cursor;
    float zoom = APP->GetZoomLevel();

    cursor = mod->GetBrushCursor(ROUND(zoom));
    APP->Set2DViewCursor(cursor, axis);
}

void AddHandlerOIFT :: OnLeaveWindow()
{
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, axis);
}

void AddHandlerOIFT :: OnLeftClick(int p)
{
    AdjRel *A;
    int o, cod;

    o = mod->obj_sel;
    if (o < 0) return;

    mod->markerID++;
    cod = mod->GetCodeValue(mod->markerID, o + 1);
    APP->SetLabelColour(cod, mod->obj_color[o]);
    A = mod->GetBrush();
    APP->AddSeedsInBrush(p, o + 1, mod->markerID, A, axis);
    APP->DrawBrush(p, cod, A, axis);


    printf("Object name: %s and label %d and marker ID %d\n", mod->obj_name[o], o + 1, APP->GetSeedId(p));

    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
}

void AddHandlerOIFT :: OnRightClick(int p)
{
    AdjRel *A;
    int cod;

    mod->markerID++;
    cod = mod->GetCodeValue(mod->markerID, 0);
    APP->SetLabelColour(cod, NIL);
    A = mod->GetBrush();
    APP->AddSeedsInBrush(p, 0, mod->markerID, A, axis);
    APP->DrawBrush(p, cod, A, axis);

    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
}

void AddHandlerOIFT :: OnMiddleClick(int p)
{
    mod->Run();
}


void AddHandlerOIFT :: OnMouseWheel(int p, int rotation, int delta)
{
    if (rotation > 0) mod->NextBrush();
    else           mod->PrevBrush();

    wxCursor *cursor;
    float zoom = APP->GetZoomLevel();

    cursor = mod->GetBrushCursor(ROUND(zoom));
    APP->Set2DViewCursor(cursor, axis);
}


void AddHandlerOIFT :: OnLeftDrag(int p, int q)
{
    AdjRel *A;
    int o, cod;

    o = mod->obj_sel;
    if (o < 0) return;
    cod = mod->GetCodeValue(mod->markerID, o + 1);
    A   = mod->GetBrush();
    APP->AddSeedsInBrushTrace(p, q, o + 1, mod->markerID, A, axis);
    APP->DrawBrushTrace(p, q, cod, A, axis);

    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
}

void AddHandlerOIFT :: OnRightDrag(int p, int q)
{
    AdjRel *A;
    int cod;

    cod = mod->GetCodeValue(mod->markerID, 0);
    A = mod->GetBrush();
    APP->AddSeedsInBrushTrace(p, q, 0, mod->markerID,
                              A, axis);
    APP->DrawBrushTrace(p, q, cod, A, axis);


    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
}

void AddHandlerOIFT :: OnLeftRelease(int p)
{
}

void AddHandlerOIFT :: OnRightRelease(int p)
{
}


void AddHandlerOIFT :: OnMouseMotion(int p)
{
    wxCursor *cursor = NULL;
    static float zoom = -1.0;

    if (zoom != APP->GetZoomLevel())
    {
        zoom = APP->GetZoomLevel();
        cursor = mod->GetBrushCursor(ROUND(zoom));
        APP->Set2DViewCursor(cursor, axis);
    }

    APP->Window->SetStatusText(_T("Mouse Left: Add Object Marker, Mouse Center: Run, Mouse Right: Add Background Marker"));
}

} //end OIFT namespace
