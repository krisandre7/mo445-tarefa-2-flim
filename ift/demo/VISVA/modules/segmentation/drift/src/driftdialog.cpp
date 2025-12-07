
#include "driftdialog.h"


#include "../xpm/del.xpm"
#include "../xpm/add.xpm"
#include "../xpm/nav.xpm"
#include "../xpm/livemarkers.xpm"

#include "../xpm/eye.xpm"
#include "../xpm/noteye.xpm"
#include "../xpm/trash.xpm"

namespace DRIFT
{

BEGIN_EVENT_TABLE(DRIFTDialog, BaseDialog)
    EVT_SIZE    (       DRIFTDialog::OnSize)
END_EVENT_TABLE()


DRIFTDialog :: DRIFTDialog(wxWindow *parent, ModuleDRIFT *mod): BaseDialog(parent, (char *)"Diff. Relaxed IFT")
{
    wxBitmap *bm[4];
    wxSize size(240, 380);
    wxColour wxcolor;

    this->SetMinSize(size);
    this->SetSize(size);
    this->mod = mod;

    panel = new BasePanel(this);
    panel->Show(true);

    xhandler = new AddHandlerDRIFT('x', mod);
    yhandler = new AddHandlerDRIFT('y', mod);
    zhandler = new AddHandlerDRIFT('z', mod);
    dhandler = new DelHandlerDRIFT(mod);

    live_xhandler = new LiveHandlerDRIFT('x', mod);
    live_yhandler = new LiveHandlerDRIFT('y', mod);
    live_zhandler = new LiveHandlerDRIFT('z', mod);

    bm[0] = new wxBitmap(nav_xpm);
    bm[1] = new wxBitmap(add_xpm);
    bm[2] = new wxBitmap(del_xpm);
    bm[3] = new wxBitmap(livemarkers_xpm);

    id_but  = APP->idManager->AllocID();
    id_bp   = APP->idManager->AllocID();
    id_res  = APP->idManager->AllocID();
    id_run  = APP->idManager->AllocID();
    id_add  = APP->idManager->AllocID();
    id_relax = APP->idManager->AllocID();
    mod->id_spin = APP->idManager->AllocID();
    mod->id_check = APP->idManager->AllocID();

    but = new BitmapRadioButton(panel, id_but, bm, 4);
    but->SetSelection(1);

    size.SetHeight(40);
    size.SetWidth(70);
    res = new wxButton(panel, id_res, _T("Reset"), wxDefaultPosition,
                       size, wxBU_EXACTFIT,
                       wxDefaultValidator, _T("butReset"));
    addobj = new wxButton(panel, id_add, _T("Add"), wxDefaultPosition,
                        size, wxBU_EXACTFIT,
                        wxDefaultValidator, _T("butAdd"));
    run = new wxButton(panel, id_run, _T("Run"), wxDefaultPosition,
                       size, wxBU_EXACTFIT,
                       wxDefaultValidator, _T("butRun"));
    relax = new wxButton(panel, id_relax, _T("Relax"), wxDefaultPosition,
                         size, wxBU_EXACTFIT,
                         wxDefaultValidator, _T("butRelax"));
    mod->spin = new wxSpinCtrl(panel, mod->id_spin, wxEmptyString, wxDefaultPosition,
                               wxDefaultSize, wxSP_ARROW_KEYS | wxSP_VERTICAL, 0, 0, 0, _T("RelaxIterations"));
    
    mod->checksmooth = new wxCheckBox(panel, mod->id_check, _T("Exec. Smooth"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("Smooth_onoff"));
    mod->checksmooth->SetValue(true);
    mod->checksmooth->Show(true);

    mod->spin->SetRange(0, 100);
    mod->spin->SetValue(5);
    /* When enter is pressed, run button will be executed */
    run->SetDefault();

    SetColor(&wxcolor, 0xff0000);
    res->SetBackgroundColour(wxcolor);
    SetColor(&wxcolor, 0xffffff);
    res->SetForegroundColour(wxcolor);
    SetColor(&wxcolor, 0xffff00);
    addobj->SetBackgroundColour(wxcolor);
    SetColor(&wxcolor, 0x00ff00);
    run->SetBackgroundColour(wxcolor);
    SetColor(&wxcolor, 0x00CCFF);
    relax->SetBackgroundColour(wxcolor);

    bPicker = new BrushPicker(panel, id_bp, true);
    wxStaticText *tbp = new wxStaticText(panel, -1, _T("Brush"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText1"));

    wxBoxSizer *hbs2 = new wxBoxSizer(wxHORIZONTAL);
    hbs2->Add(tbp,     0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);
    hbs2->Add(bPicker, 0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);

    wxBoxSizer *hbs3 = new wxBoxSizer(wxHORIZONTAL);
    hbs3->Add(res,  0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);
    hbs3->Add(addobj, 0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);
    hbs3->Add(run,  0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);
    hbs3->Add(relax,  0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);

    //Horizontal box sizer 4. This groups the spin and checkbox
    wxBoxSizer *hbs4 = new wxBoxSizer(wxHORIZONTAL);
    hbs4->Add(mod->spin,  0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);
    hbs4->Add(mod->checksmooth, 0, wxALIGN_CENTER | wxALIGN_CENTER_VERTICAL);

    wxStaticBoxSizer *tools_sizer;
    tools_sizer = new wxStaticBoxSizer(wxVERTICAL, panel, _T("Tools"));
    wxStaticBoxSizer *actions_sizer;
    actions_sizer = new wxStaticBoxSizer(wxVERTICAL, panel, _T("Actions"));
    wxStaticText *tri = new wxStaticText(panel, -1, _T("Relax Iterations"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText1"));

    tools_sizer->AddSpacer(10);
    tools_sizer->Add(tri, 0, wxALIGN_CENTER);
    tools_sizer->Add(hbs4, 0, wxALIGN_CENTER);
    // tools_sizer->Add(mod->checksmooth, 0, wxALIGN_CENTER);
    // tools_sizer->Add(mod->spin, 0, wxALIGN_CENTER);
    tools_sizer->Add(but,    0, wxALIGN_CENTER);
    tools_sizer->Add(hbs2,   0, wxALIGN_CENTER);

    actions_sizer->Add(hbs3,   0, wxALIGN_CENTER);

    panel->sizer->Add(actions_sizer, 0, wxEXPAND);
    panel->sizer->Add(tools_sizer,   0, wxEXPAND);
    panel->sizer->AddSpacer(20);

    panel->sizer->SetSizeHints(panel);
    panel->sizer->Layout();

    this->AddPanel(panel, 0);
    //----------------------------------
    size.SetHeight(30);
    size.SetWidth(60);
    //addobj = new wxButton(this, id_add, _T("Add"), wxDefaultPosition, size, wxBU_EXACTFIT, wxDefaultValidator, _T("butAdd"));
    //addobj = new wxButton(this, id_add, _T("Add"), wxDefaultPosition, size, wxBU_EXACTFIT, wxDefaultValidator, _T("butAdd"));
    //this->sizer->Add(addobj, 0, wxALIGN_RIGHT);
    //this->sizer->Prepend(addobj, 0, wxALIGN_RIGHT);


    //----------ObjectPanel-------------
    objPanel = this->CreateObjectPanel();
    this->AddPanel((wxPanel *)objPanel);

    ChangeObjSelection(v_panel_bkg[mod->obj_sel]);
    //----------------------------------

    Connect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
             wxCommandEventHandler(DRIFTDialog::OnChangeMode),
             NULL, NULL );
    Connect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
             wxCommandEventHandler(DRIFTDialog::OnReset),
             NULL, NULL );
    Connect( id_run, wxEVT_COMMAND_BUTTON_CLICKED,
             wxCommandEventHandler(DRIFTDialog::OnRun),
             NULL, NULL );
    Connect( id_add, wxEVT_COMMAND_BUTTON_CLICKED,
             wxCommandEventHandler(DRIFTDialog::OnAddObj),
             NULL, NULL );
    Connect( id_relax, wxEVT_COMMAND_BUTTON_CLICKED,
             wxCommandEventHandler(DRIFTDialog::OnRelax),
             NULL, NULL );


    //Just for the add be the default
    APP->Set2DViewInteractionHandler(xhandler, 'x');
    APP->Set2DViewInteractionHandler(yhandler, 'y');
    APP->Set2DViewInteractionHandler(zhandler, 'z');
}


DRIFTDialog::~DRIFTDialog()
{
    Disconnect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
                wxCommandEventHandler(DRIFTDialog::OnChangeMode),
                NULL, NULL );
    Disconnect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
                wxCommandEventHandler(DRIFTDialog::OnReset),
                NULL, NULL );
    Disconnect( id_relax, wxEVT_COMMAND_BUTTON_CLICKED,
                wxCommandEventHandler(DRIFTDialog::OnRelax),
                NULL, NULL );
    Disconnect( id_run, wxEVT_COMMAND_BUTTON_CLICKED,
                wxCommandEventHandler(DRIFTDialog::OnRun),
                NULL, NULL );
    Disconnect( id_add, wxEVT_COMMAND_BUTTON_CLICKED,
                wxCommandEventHandler(DRIFTDialog::OnAddObj),
                NULL, NULL );
    APP->idManager->FreeID(id_but);
    APP->idManager->FreeID(id_bp);
    APP->idManager->FreeID(id_res);
    APP->idManager->FreeID(id_relax);
    APP->idManager->FreeID(id_run);
    APP->idManager->FreeID(id_add);
    APP->SetDefaultInteractionHandler();

    delete xhandler;
    delete yhandler;
    delete zhandler;
    delete dhandler;

    delete live_xhandler;
    delete live_yhandler;
    delete live_zhandler;

    free(v_panel_bkg);
    free(v_but_color);
    free(v_but_eye);
    free(v_but_trash);
}

wxScrolledWindow *DRIFTDialog::CreateObjectPanel()
{
    int i, n, id, w, h, wm = 200;
    wxSize size(220, 120);
    wxPanel *bkg;

    wxScrolledWindow *swind = new wxScrolledWindow(this, wxID_ANY,
            wxDefaultPosition,
            wxDefaultSize,
            wxHSCROLL | wxVSCROLL,
            _T("scrolledWindow"));
    swind->SetMinSize(size);
    wxBitmap *bmeye   = new wxBitmap(eye_xpm);
    wxBitmap *bmneye  = new wxBitmap(noteye_xpm);
    wxBitmap *bmtrash = new wxBitmap(trash_xpm);

    n = mod->nobjs;
    if (n > 0)
    {
        v_panel_bkg = (wxPanel **)malloc(sizeof(wxPanel *)*n);
        v_but_color = (AlphaColourButton **)malloc(sizeof(AlphaColourButton *)*n);
        v_but_eye = (BitmapToggleButton **)malloc(sizeof(BitmapToggleButton *)*n);
        v_but_trash = (wxBitmapButton **)malloc(sizeof(wxBitmapButton *)*n);
    }
    else
    {
        v_panel_bkg = NULL;
        v_but_color = NULL;
        v_but_eye   = NULL;
        v_but_trash = NULL;
    }

    swind->SetScrollRate(5, 5);
    swind->Scroll(0, 0);
    swind->SetVirtualSize(200, 10 + MAX(40 * n, 40));

    for (i = 0; i < n; i++)
    {
        size.SetHeight(wxDefaultCoord);
        size.SetWidth(200);
        bkg = new ObjBkgPanel(swind,
                              wxPoint(10, 10 + 40 * i),
                              wxDefaultSize,
                              this);
        //bkg->SetSize(size);
        bkg->SetMinSize(size);
        v_panel_bkg[i] = bkg;
        v_but_color[i] = new AlphaColourButton(bkg, wxID_ANY);
        v_but_eye[i] = new BitmapToggleButton(bkg, wxID_ANY,
                                              bmneye, bmeye);
        v_but_trash[i] = new wxBitmapButton(bkg, wxID_ANY, *bmtrash,
                                            wxDefaultPosition, wxDefaultSize,
                                            wxBU_AUTODRAW, wxDefaultValidator,
                                            _T("trash"));
        v_but_color[i]->SetValue(mod->obj_color[i]);
        v_but_color[i]->SetAlpha(255);
        v_but_eye[i]->SetValue(mod->obj_visibility[i]);

        id = v_but_trash[i]->GetId();
        Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(DRIFTDialog::OnDelete),
                 NULL, NULL );

        id = v_but_color[i]->GetId();
        Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(DRIFTDialog::OnChangeColor),
                 NULL, NULL );

        id = v_but_eye[i]->GetId();
        Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                 wxCommandEventHandler(DRIFTDialog::OnChangeVisibility),
                 NULL, NULL );

        wxString *wxstr = new wxString(mod->obj_name[i], wxConvUTF8);
        ObjLabel *tname = new ObjLabel(bkg, *wxstr, this);
        size.SetHeight(wxDefaultCoord);
        size.SetWidth(120);
        //tname->SetSize(size);
        tname->SetMinSize(size);
        wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
        hsizer->Add(v_but_eye[i],   0, wxALIGN_LEFT | wxEXPAND);
        hsizer->Add(v_but_trash[i], 0, wxALIGN_RIGHT | wxEXPAND);
        hsizer->Add(v_but_color[i], 0, wxALIGN_LEFT | wxEXPAND);
        hsizer->AddSpacer(10);
        hsizer->Add(tname,          1, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);

        bkg->SetSizer(hsizer, true);
        hsizer->SetSizeHints(bkg);
        hsizer->Layout();

        bkg->GetSize(&w, &h);
        if (w > wm) wm = w;
    }
    for (i = 0; i < n; i++)
    {
        size.SetHeight(wxDefaultCoord);
        size.SetWidth(wm);
        (v_panel_bkg[i])->SetSize(size);
    }
    swind->SetVirtualSize(wm, 10 + MAX(40 * n, 40));

    wxPaintDC paintDC(swind);
    swind->DoPrepareDC(paintDC);

    delete bmeye;
    delete bmneye;
    delete bmtrash;

    return (swind);
}


void DRIFTDialog::DestroyObjectPanel()
{
    int i, n, id;

    n = mod->nobjs;
    for (i = 0; i < n; i++)
    {
        if (v_but_trash == NULL) break;
        id = v_but_trash[i]->GetId();
        Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(DRIFTDialog::OnDelete),
                    NULL, NULL );
        if (v_but_color == NULL) break;
        id = v_but_color[i]->GetId();
        Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(DRIFTDialog::OnChangeColor),
                    NULL, NULL );
        if (v_but_eye == NULL) break;
        id = v_but_eye[i]->GetId();
        Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
                    wxCommandEventHandler(DRIFTDialog::OnChangeVisibility),
                    NULL, NULL );
    }
    if (v_panel_bkg != NULL) free(v_panel_bkg);
    if (v_but_color != NULL) free(v_but_color);
    if (v_but_eye != NULL)   free(v_but_eye);
    if (v_but_trash != NULL) free(v_but_trash);
    v_panel_bkg = NULL;
    v_but_color = NULL;
    v_but_eye   = NULL;
    v_but_trash = NULL;

    if (objPanel != NULL)
    {
        if (sizer->GetItem(objPanel, false) != NULL)
            sizer->Detach(objPanel);
        objPanel->Destroy();
        objPanel = NULL;
    }
}


void DRIFTDialog :: OnAddObj(wxCommandEvent &event)
{
    wxSize size(280, 380);
    //wxSize size(800, 600);
    int i, r, n, o;
    char msg[500];

    if (mod->nobjs >= MAX_OBJS) return;

    n = (APP->Data.label)->n;
    i = mod->nobjs;
    mod->obj_visibility[i] = true;
    r = APP->ShowNewObjDialog(&mod->obj_color[i], mod->obj_name[i]);
    // printf("Object label: %d\n", i);
    // printf("Chosen color: %d\n", mod->obj_color[i]);
    // printf("Chosen name: %s\n", mod->obj_name[i]);
    //APP->SetLabelColour(i+1, mod->obj_color[i]);
    if (r != 0) return;
    //---------------------------
    for (o = 0; o < mod->nobjs; o++)
    {
        if (strcmp(mod->obj_name[i], mod->obj_name[o]) == 0 || strcasecmp(mod->obj_name[i], "All") == 0)
        {
            sprintf(msg, "You are already segmenting the '%s'."\
                    "\nChoose a different object.",
                    mod->obj_name[i]);
            wxString wxmsg_w(msg, wxConvUTF8);
            wxMessageBox(wxmsg_w, _T("Warning"),
                         wxOK | wxICON_EXCLAMATION, APP->Window);
            return;
        }
    }
    //---------------------------
    this->DestroyObjectPanel();

    mod->nobjs++;
    mod->obj_sel = i;
    if (mod->obj_sel < 0) mod->obj_sel = 0;

    APP->SetLabelColour(i+1, mod->obj_color[i]);

    this->SetMinSize(size);
    this->SetSize(size);
    objPanel = this->CreateObjectPanel();
    this->AddPanel((wxPanel *)objPanel);
    ChangeObjSelection(v_panel_bkg[mod->nobjs - 1]); // last one
    this->Fit();
    this->SetMinSize(size);
    this->SetSize(size);
    this->Update();
}


void DRIFTDialog :: OnDelete(wxCommandEvent &event)
{
    wxSize size(280, 380);
    char msg[1024];
    int i, n, id;

    this->UnmarkAllForRemoval();

    id = event.GetId();
    n = mod->nobjs;
    for (i = 0; i < n; i++)
    {
        if (id == v_but_trash[i]->GetId())
            break;
    }
    if (i < n)
    {
        sprintf(msg, "Do you really wish to delete the %s?", mod->obj_name[i]);

        wxString wxmsg(msg, wxConvUTF8);
        wxMessageDialog *dialog = new wxMessageDialog(this, wxmsg,
                _T("Delete Confirmation"),
                wxYES_NO | wxICON_QUESTION,
                wxDefaultPosition);
        if (dialog->ShowModal() == wxID_YES)
        {
            this->DestroyObjectPanel();
            mod->DeleteObj(i);

            this->SetMinSize(size);
            this->SetSize(size);
            objPanel = this->CreateObjectPanel();
            this->AddPanel((wxPanel *)objPanel);
            if (mod->obj_sel >= 0 && v_panel_bkg != NULL)
                ChangeObjSelection(v_panel_bkg[mod->obj_sel]);
            //this->Fit();
            this->SetMinSize(size);
            this->SetSize(size);
            this->Update();
        }
    }
}


void DRIFTDialog :: UnmarkAllForRemoval()
{
    int Cmax, cod, o;
    APP->UnmarkAllForRemoval();

    Cmax = MaximumValue3(APP->Data.label);
    for (cod = 0; cod <= Cmax; cod++)
    {
        o = mod->GetCodeLabel(cod);
        if (o == 0)
            APP->SetLabelColour(cod, NIL);
        else if (mod->obj_visibility[o - 1])
            APP->SetLabelColour(cod, mod->obj_color[o - 1]);
        else
            APP->SetLabelColour(cod, NIL);
    }
}


void DRIFTDialog :: OnChangeColor(wxCommandEvent &event)
{
    int i, n, id, Cmax, cod;

    this->UnmarkAllForRemoval();

    id = event.GetId();
    n = mod->nobjs;
    for (i = 0; i < n; i++)
    {
        if (id == v_but_color[i]->GetId())
            break;
    }
    if (i < n)
    {
        mod->obj_color[i] = v_but_color[i]->GetValue();

        Cmax = MaximumValue3(APP->Data.label);
        for (cod = 0; cod <= Cmax; cod++)
        {
            if (mod->GetCodeLabel(cod) == i + 1)
                APP->SetLabelColour(cod, mod->obj_color[i]);
        }
        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);
    }
}


void DRIFTDialog :: OnChangeVisibility(wxCommandEvent &event)
{
    int i, n, id, Cmax, cod;

    this->UnmarkAllForRemoval();

    id = event.GetId();
    n = mod->nobjs;
    for (i = 0; i < n; i++)
    {
        if (id == v_but_eye[i]->GetId())
            break;
    }
    if (i < n)
    {
        mod->obj_visibility[i] = v_but_eye[i]->GetValue();

        Cmax = MaximumValue3(APP->Data.label);
        for (cod = 0; cod <= Cmax; cod++)
        {
            if (mod->GetCodeLabel(cod) == i + 1)
            {
                if (mod->obj_visibility[i])
                    APP->SetLabelColour(cod, mod->obj_color[i]);
                else
                    APP->SetLabelColour(cod, NIL);
            }
        }
        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);
    }
}


void DRIFTDialog::OnChangeMode(wxCommandEvent &event)
{
    DRIFTDialog::ModeType mode = GetOperationMode();

    if (mode == DRIFTDialog::ADDMARKER)
    {
        APP->Set2DViewInteractionHandler(xhandler, 'x');
        APP->Set2DViewInteractionHandler(yhandler, 'y');
        APP->Set2DViewInteractionHandler(zhandler, 'z');
    }
    else if (mode == DRIFTDialog::DELMARKER)
    {
        APP->Set2DViewInteractionHandler(dhandler, 'x');
        APP->Set2DViewInteractionHandler(dhandler, 'y');
        APP->Set2DViewInteractionHandler(dhandler, 'z');
    }
    else if (mode == DRIFTDialog::LIVEMARKER)
    {
        APP->Set2DViewInteractionHandler(live_xhandler, 'x');
        APP->Set2DViewInteractionHandler(live_yhandler, 'y');
        APP->Set2DViewInteractionHandler(live_zhandler, 'z');
    }
    else
    {
        APP->SetDefaultInteractionHandler();
    }
    panel->sizer->Layout();
}


DRIFTDialog::ModeType DRIFTDialog::GetOperationMode()
{
    int sel = but->GetSelection();

    if (sel == 0)      return DRIFTDialog::NAVIGATOR;
    else if (sel == 1) return DRIFTDialog::ADDMARKER;
    else if (sel == 2) return DRIFTDialog::DELMARKER;
    else if (sel == 3) return DRIFTDialog::LIVEMARKER;
    else            return DRIFTDialog::NAVIGATOR;
}


void DRIFTDialog::OnReset(wxCommandEvent &WXUNUSED(event))
{
    mod->Reset();
}


void DRIFTDialog::OnRun(wxCommandEvent &WXUNUSED(event))
{
    mod->Run();
}

void DRIFTDialog::OnRelax(wxCommandEvent &WXUNUSED(event))
{
    mod->Relax();
}


/*
void DRIFTDialog::OnFinish(wxCommandEvent& WXUNUSED(event)){
  mod->Finish();
}
*/

AdjRel *DRIFTDialog :: GetBrush()
{
    return bPicker->GetBrush();
}

wxCursor *DRIFTDialog :: GetBrushCursor(int zoom)
{
    return bPicker->GetBrushCursor(zoom);
}

void DRIFTDialog :: NextBrush()
{
    bPicker->NextBrush();
}

void DRIFTDialog :: PrevBrush()
{
    bPicker->PrevBrush();
}


void DRIFTDialog :: OnCancel(wxCommandEvent &event)
{
    mod->Stop();
    //BaseDialog::OnCancel(event);
}

void DRIFTDialog :: OnOk(wxCommandEvent &event)
{
    mod->Finish();
}


void DRIFTDialog :: OnSize(wxSizeEvent &event)
{
    BaseDialog::OnSize(event);
}

void DRIFTDialog :: ChangeObjSelection(wxPanel *objbkg)
{
    wxPanel *bkg = NULL;
    wxColour wxcolor;
    int n, i;

    n = mod->nobjs;
    for (i = 0; i < n; i++)
    {
        bkg = this->v_panel_bkg[i];
        if (bkg == objbkg)
        {
            mod->obj_sel = i;
            SetColor(&wxcolor, 0xffff00);
        }
        else if (i % 2 == 0) SetColor(&wxcolor, 0xffffff);
        else            SetColor(&wxcolor, 0xffffdd);
        bkg->SetBackgroundColour(wxcolor);
    }
}

//----------------------------------------------------

BEGIN_EVENT_TABLE(ObjBkgPanel, wxPanel)
    EVT_MOUSE_EVENTS(ObjBkgPanel::OnMouseEvent)
END_EVENT_TABLE()

ObjBkgPanel :: ObjBkgPanel(wxWindow *parent,
                             wxPoint &pos,
                             wxSize &size,
                           DRIFTDialog  *dialog)
    : wxPanel(parent, wxID_ANY, pos, size, wxTAB_TRAVERSAL, _T("panel_bkg"))
{

    this->dialog = dialog;
}

ObjBkgPanel :: ~ObjBkgPanel()
{
}

void ObjBkgPanel :: OnMouseEvent(wxMouseEvent &event)
{
    switch (event.GetButton())
    {
    case wxMOUSE_BTN_LEFT :
        if (event.GetEventType() == wxEVT_LEFT_DOWN)
        {
            dialog->ChangeObjSelection((wxPanel *)this);
        }
        break;
    case wxMOUSE_BTN_RIGHT :
        if (event.GetEventType() == wxEVT_RIGHT_DOWN)
        {
            dialog->ChangeObjSelection((wxPanel *)this);
        }
        break;
    }
}

//----------------------------------------------------

BEGIN_EVENT_TABLE(ObjLabel, wxStaticText)
    EVT_MOUSE_EVENTS(ObjLabel::OnMouseEvent)
END_EVENT_TABLE()

ObjLabel :: ObjLabel(wxWindow *parent,
                       wxString &label,
                     DRIFTDialog  *dialog)
    : wxStaticText(parent, wxID_ANY, label, wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("obj_label"))
{

    this->dialog = dialog;
}

ObjLabel :: ~ObjLabel()
{
}

void ObjLabel :: OnMouseEvent(wxMouseEvent &event)
{
    switch (event.GetButton())
    {
    case wxMOUSE_BTN_LEFT :
        if (event.GetEventType() == wxEVT_LEFT_DOWN)
        {
            dialog->ChangeObjSelection((wxPanel *)this->GetParent());
        }
        break;
    case wxMOUSE_BTN_RIGHT :
        if (event.GetEventType() == wxEVT_RIGHT_DOWN)
        {
            dialog->ChangeObjSelection((wxPanel *)this->GetParent());
        }
        break;
    }
}


} //end DRIFT namespace


