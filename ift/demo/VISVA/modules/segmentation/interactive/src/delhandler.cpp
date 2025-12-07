
#include "delhandler.h"

namespace Interactive
{

    DelHandler :: DelHandler(ModuleInteractive *mod)
    {
        this->mod = mod;
    }

    DelHandler :: ~DelHandler() {}


    void DelHandler :: OnEnterWindow()
    {
        APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'x');
        APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'y');
        APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'z');
    }

    void DelHandler :: OnLeaveWindow() {}

    void DelHandler :: OnLeftClick(int p)
    {
        int id, c, o, cod;

        cod = GetVoxel(APP->Data.label, p);

        if (APP->IsSeed(p))
        {
            id = APP->GetSeedId(p);
            o  = APP->GetSeedLabel(p) - 1;
        }
        else
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


    void DelHandler :: OnRightClick(int p) {}

    void DelHandler :: OnMiddleClick(int p)
    {
        mod->Run();
    }

    void DelHandler :: OnLeftDrag(int p, int q) {}

    void DelHandler :: OnRightDrag(int p, int q) {}

    void DelHandler :: OnMouseMotion(int p)
    {
        APP->Window->SetStatusText(_T("Mouse Left: Select Marker for Removal, Mouse Center: Run"));
    }

} //end Interactive namespace


