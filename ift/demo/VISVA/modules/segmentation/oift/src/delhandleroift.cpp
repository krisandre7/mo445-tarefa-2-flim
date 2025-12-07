
#include "delhandleroift.h"

namespace OIFT
{

DelHandlerOIFT :: DelHandlerOIFT(ModuleOIFT *mod)
{
    this->mod = mod;
}

DelHandlerOIFT :: ~DelHandlerOIFT() {}


void DelHandlerOIFT :: OnEnterWindow()
{
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'x');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'y');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'z');
}

void DelHandlerOIFT :: OnLeaveWindow() {}

void DelHandlerOIFT :: OnLeftClick(int p)
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


void DelHandlerOIFT :: OnRightClick(int p) {}

void DelHandlerOIFT :: OnMiddleClick(int p)
{
    mod->Run();
}

void DelHandlerOIFT :: OnLeftDrag(int p, int q) {}

void DelHandlerOIFT :: OnRightDrag(int p, int q) {}


void DelHandlerOIFT :: OnLeftRelease(int p) {

}
void DelHandlerOIFT :: OnMouseMotion(int p)
{
    APP->Window->SetStatusText(_T("Mouse Left: Select Marker for Removal, Mouse Center: Run"));
}

} //end OIFT namespace



