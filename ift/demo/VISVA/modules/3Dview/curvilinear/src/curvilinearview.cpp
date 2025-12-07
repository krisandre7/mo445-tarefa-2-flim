
#include "curvilinearview.h"

#include "../xpm/glasses.xpm"
#include "../xpm/save_canvas.xpm"

namespace Curvilinear{

    CurvilinearView :: CurvilinearView(wxWindow *parent)
            : BasePanel(parent){
        int id1,id2;

        canvas  = new CurvilinearCanvas(this);

        id1 = APP->idManager->AllocID();
        id2 = APP->idManager->AllocID();

        sDist = new wxSlider(this, id1, 0, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS, wxDefaultValidator, _T("slider1"));
        sDist->Disable();

        wxBitmap *bm = new wxBitmap(glasses_xpm);
        but3D = new BitmapToggleButton(this, id2, bm, bm);
        but3D->SetBkgColor(true, 0x808080);
        but3D->SetValue(false);

        wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
        hsizer->AddSpacer(5);
        hsizer->Add(but3D, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
        hsizer->Add(sDist, 1, wxALIGN_LEFT|wxEXPAND);

        int save_id = APP->idManager->AllocID();
        wxBitmap *bmSave = new wxBitmap(save_canvas_xpm);
        butSave = new wxBitmapButton(this, save_id, *bmSave, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("butSaveCanvas"));
        hsizer->Add(butSave, 0, wxALIGN_LEFT);
        hsizer->AddSpacer(20);


        sizer->Add(canvas, 1, wxEXPAND);
        sizer->Add(hsizer, 0, wxALIGN_LEFT|wxEXPAND);

        SetSizer(sizer, true);
        sizer->SetSizeHints(this);
        sizer->Layout();

        Connect( id1, wxEVT_SCROLL_THUMBRELEASE,
                 wxScrollEventHandler(CurvilinearView::OnChangeDistance),
                 NULL, NULL );
        Connect( id1, wxEVT_SCROLL_THUMBTRACK,
                 wxScrollEventHandler(CurvilinearView::OnChangeDistance),
                 NULL, NULL );
        Connect( id2, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(CurvilinearView::OnChange3D),
                 NULL, NULL );

        Connect( save_id, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(CurvilinearView::OnSaveCanvasView),
                 NULL, NULL );

    }


    CurvilinearView :: ~CurvilinearView(){
        int id1 = sDist->GetId();
        int id2 = but3D->GetId();

        Disconnect( id1, wxEVT_SCROLL_THUMBRELEASE,
                    wxScrollEventHandler(CurvilinearView::OnChangeDistance),
                    NULL, NULL );
        Disconnect( id1, wxEVT_SCROLL_THUMBTRACK,
                    wxScrollEventHandler(CurvilinearView::OnChangeDistance),
                    NULL, NULL );
        Disconnect( id2, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(CurvilinearView::OnChange3D),
                    NULL, NULL );
        APP->idManager->FreeID(id1);
        APP->idManager->FreeID(id2);
    }


    void CurvilinearView :: CurvilinearRefresh(bool dataChanged,
                                               float quality){
        CurvilinearCanvas *scanvas;
        int skip;

        if(!APP->Data.loaded) return;

        sDist->Enable(true);
        scanvas = (CurvilinearCanvas *)this->canvas;
        skip = ROUND((1.0-quality)*11)+1;
        scanvas->drawRender(dataChanged, skip);
    }


    int CurvilinearView :: GetDistance(){
        float dx = (APP->Data.orig)->dx;
        //Convert from mm to pixels.
        return ROUND( sDist->GetValue()/dx );
    }


    void CurvilinearView :: OnChangeDistance(wxScrollEvent& WXUNUSED(event)){
        APP->Refresh3DCanvas(false, 1.0);
    }

    void CurvilinearView :: OnChange3D(wxCommandEvent& event){
        APP->Refresh3DCanvas(false, 1.0);
    }

    void CurvilinearView :: OnSaveCanvasView(wxCommandEvent & event) {
        wxString path;

        path = wxSaveFileSelector(_("Save canvas view"),_(""), _(""), APP->Window);

        if(path != _("")) {
            wxImage *img = canvas->GetImage();

            if(!img->SaveFile(path))
                wxMessageBox(_("Invalid file extension"), _("Error saving image"));
        }
    }

} //end Curvilinear namespace

