
#include "delhandlerdrift.h"

namespace DRIFT
{

DelHandlerDRIFT :: DelHandlerDRIFT(ModuleDRIFT *mod)
{
    this->mod = mod;
}

DelHandlerDRIFT :: ~DelHandlerDRIFT() {}


void DelHandlerDRIFT :: OnEnterWindow()
{
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'x');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'y');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'z');
}

void DelHandlerDRIFT :: OnLeaveWindow() {}

void DelHandlerDRIFT :: OnLeftClick(int p)
{
    int id, c, o, cod;

    cod = GetVoxel(APP->Data.label, p);

    if (APP->IsSeed(p)) // The click was on the marker
    {
        id = APP->GetSeedId(p);
        o  = APP->GetSeedLabel(p) - 1;
    }
    else // The click was on a marker region
    {
        id = mod->GetCodeID(cod);
        o  = mod->GetCodeLabel(cod) - 1;
    }

    if (id == 0) return;
    c = APP->GetLabelColour(cod);
    if (APP->IsIdMarkedForRemoval(id))
    {

        APP->UnmarkForRemoval(id);
        if (o >= 0 && c == inverseColor(mod->obj_color[o]))
            APP->SetLabelColour(cod, mod->obj_color[o]);
        else
            APP->SetLabelColour(cod, NIL);
    }
    else
    {
        APP->MarkForRemoval(id);
        if (o < 0 && c == NIL)
            APP->SetLabelColour(cod, 0x0000aa);
        else
            APP->SetLabelColour(cod, inverseColor(mod->obj_color[o]));
    }

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(false, 1.0);
}


void DelHandlerDRIFT :: OnRightClick(int p) {}

void DelHandlerDRIFT :: OnMiddleClick(int p)
{
    mod->Run();
}

void DelHandlerDRIFT :: OnLeftDrag(int p, int q) {}

void DelHandlerDRIFT :: OnRightDrag(int p, int q) {}


void DelHandlerDRIFT :: OnLeftRelease(int p) {

}
void DelHandlerDRIFT :: OnMouseMotion(int p)
{
    APP->Window->SetStatusText(_T("Mouse Left: Select Marker for Removal, Mouse Center: Run"));
}

} //end DRIFT namespace



