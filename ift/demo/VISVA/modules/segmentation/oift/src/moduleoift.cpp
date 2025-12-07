#include "moduleoift.h"
#include "oiftdialog.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace OIFT
{

    ModuleOIFT :: ModuleOIFT()
            : SegmentationModule()
    {
        SetName((char *)"Oriented IFT");
        SetAuthor((char *)"Thiago Vallin Spina");
        SoftwareVersion ver(1, 0, 0);
        SetVersion(ver);

        this->InitNull();

        optDialog = NULL;
        this->alpha = 0.04;
        this->beta = 12;
        this->gamma = 3;
        this->niters = 3;

        // Robot-based reconstruction
        this->ori_watershed_gamma = 4;
        this->nseeds_per_label_per_iteration = 4;
        this->min_safe_distance_to_border = 2;
        this->max_marker_width = 2;
        this->max_marker_length = 20;
        this->stopping_threshold = 0.995;
        this->sec_stopping_threshold = 10000000;

        obj_visibility[0] = true;
        obj_color[0] = 0xffff00;
        obj_name[0][0] = '\0';
        obj_sel = 0;
        nobjs = 0;
    }


    ModuleOIFT :: ~ModuleOIFT()
    {

    }

    void ModuleOIFT :: Start()
    {
        static const char *title = "Resume segmentation?";
        static const char *msg = "The selected object already exists.\nDo you want to resume the previous segmentation?\n ('Yes' to resume, 'No' to start from scratch).";
        SegmObject *obj = NULL;
        Scene *bin = NULL;
        Set *S = NULL, *aux;
        int r, p, id, cod;
        int resume = 0;
        int number_of_objects = 0;
        timer tic;
        wxString resuming_type = _("None");
        iftLabeledSet *segm_seeds = NULL, *tmp = NULL;
        iftImage *reconstructed_label = NULL;

        this->markerID = 0;
        APP->EnableObjWindow(false);
        if (APP->Data.loaded)
        {
            number_of_objects = MaximumValue3(APP->Data.label);
            if (number_of_objects > 0)
            {
//----- Ask to resume the segmentation ------------
                wxString wxtitle(title, wxConvUTF8);
                wxString wxmsg(msg, wxConvUTF8);

                wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);

                if (dialog.ShowModal() != wxID_YES)
                    resume = 0;
                else
                    resume = 1;

                if(resume) {
                    wxArrayString resuming_choices;

                    resuming_choices.Add(_("ISF"));
                    resuming_choices.Add(_("Robot"));

                    resuming_type = wxGetSingleChoice(_("Select the resuming type:"), _("Resuming type"), resuming_choices, APP->Window);
                }
            }
        }


        if (!resume)
        {
            iftAdjRel *A = iftSpheric(1.0);

            APP->ResetData();
            r = APP->ShowNewObjDialog(&this->obj_color[0], this->obj_name[0]);
            if (r != 0) return;

            this->active = true;
            this->nobjs = 1;
            this->obj_sel = 0;
            APP->SetLabelColour(0, NIL);
            APP->SetLabelColour(1, this->obj_color[0]);

            this->AllocBasicData();

//            iftIGraphResetWatershed(this->igraph, this->Q);
            this->ComputeGradientImageBasins(A);

            this->fst = iftCreateImageForest(this->gradient, A);

            this->DisplayPanel();
        }
        else
        {
            this->active = true;
            this->nobjs = number_of_objects;
            this->obj_sel = 0;
            APP->SetLabelColour(0, NIL);

            if(APP->GetNumberOfObjs() > MAX_OBJS) {
                wxMessageBox(_("Cannot allocate data for that many objects"), _("Memory allocation error"),
                             wxICON_ERROR);

            } else {

                if (APP->GetNumberOfObjs() == number_of_objects) {
                    SegmObject *aux = NULL;
                    for (int i = 0; i < number_of_objects; i++) {
                        aux = APP->GetObjByIndex(i);
                        strcpy(this->obj_name[i], aux->name);
                        this->obj_color[i] = aux->color;
                        this->obj_visibility[i] = true;
                    }
                }


//----- resuming previous segmentation: ------------


                this->markerID = 0;

                if (resuming_type == _("Robot")) {
                    iftAdjRel *A = iftSpheric(1.0);

                    this->AllocBasicData();

                    this->del_alg_type = ModuleOIFT::OIFTResuming;

                    this->DisplayParamPanel();

                    this->ComputeGradientImageBasins(A);

                    this->fst = iftCreateImageForest(this->gradient, A);

                    this->DisplayPanel();

                    APP->Busy((char *) "Please wait, resuming previous segmentation...");
                    APP->StatusMessage((char *) "Please wait - Resuming Segmentation...");

//                    iftIGraphResetWatershed(this->igraph, this->Q);

                    iftDestroyImage(&this->presegmentation);
                    this->presegmentation = APP->SceneToiftImage(APP->Data.label);



                    segm_seeds = iftLabelToForestPixelRobotEDTOrientedWatershed(this->fst,
                                                                                this->presegmentation, nseeds_per_label_per_iteration,
                                                                                min_safe_distance_to_border, max_marker_width,
                                                                                max_marker_length, ori_watershed_gamma, NULL,
                                                                                stopping_threshold, sec_stopping_threshold,
                                                                                iftStopLabelReconstructionByDice,
                                                                                iftStopLabelReconstructionByNumInterations,
                                                                                true);
                    reconstructed_label = iftCopyImage(this->fst->label);

                } else {
                    int nseeds;

                    this->AllocBasicData();
                    this->del_alg_type = ModuleOIFT::ISFResuming;

                    this->DisplayParamPanel();

                    this->DisplayPanel();

                    APP->Busy((char *) "Please wait, resuming previous segmentation...");
                    APP->StatusMessage((char *) "Please wait - Resuming Segmentation...");

                    iftDestroyImage(&this->presegmentation);
                    this->presegmentation = APP->SceneToiftImage(APP->Data.label);


                    segm_seeds = iftLabelToForestISF_Root(this->iftorig, this->presegmentation, this->alpha, this->beta,
                                                          this->gamma, this->niters, &nseeds, &this->igraph, &this->Q);

                    this->ComputeGradientImageBasins(this->igraph->A);


                    iftIGraphSetWeight(this->igraph, this->gradient);
                    reconstructed_label = iftIGraphLabel(igraph);
                }

                for (iftLabeledSet *s = segm_seeds; s != NULL; s = s->next) {
                    int p, label, id, cod;

                    p = s->elem;

                    label = s->label;
                    id = s->marker; // A unique marker ID must be set for all label reconstruction procedures

                    // Updating the marker ID to match the one that was last added by the resuming method
                    this->markerID = iftMax(this->markerID, id);

                    cod = this->GetCodeValue(id, label);

                    // Adding a new seed
                    APP->AddSeed(p, label, id);

                    if (label == 0)
                        APP->SetLabelColour(cod, NIL);
                    else
                        APP->SetLabelColour(cod, this->obj_color[label - 1]);
                }



                iftDblArray *error = iftDiceSimilarityMultiLabel(this->presegmentation, reconstructed_label,
                                                                 number_of_objects);
                printf("Dice: %f\n", error->val[0]);
                printf("Maximum marker ID: %d\n", this->markerID);
                iftDestroyDblArray(&error);
                iftDestroyImage(&reconstructed_label);

                iftDestroyLabeledSet(&segm_seeds);

                if(this->igraph != NULL) {
                    // NOTE: convert label here, because the seeds must be set before it
                    APP->Data.label = APP->IGraphLabelImageToScene(this->igraph);
                } else {
                    APP->Data.label = APP->LabelImageToScene(this->fst);
                }


                APP->StatusMessage((char *)"Done");
                APP->Unbusy();
            }

            APP->Refresh2DCanvas();
            APP->Refresh3DCanvas(true, 1.0);
        }
    }

    void ModuleOIFT :: InitNull()
    {
        this->iftorig = NULL;
        this->gradient = NULL;

        this->igraph = NULL;
        this->presegmentation = NULL;
        this->Q = NULL;

        this->fst = NULL;


        this->execution = IFT_NIL;
    }

    void ModuleOIFT :: ComputeGradientImageBasins(iftAdjRel *A)
    {
        timer *t1, *t2;

        if (MaximumValue3(APP->Data.arcw) == 0)
            DestroyScene(&(APP->Data.arcw));

        if (this->gradient != NULL) iftDestroyImage(&this->gradient);

        if (APP->Data.arcw == NULL)
        {
            APP->Busy((char *)"Please wait, computing gradient...");
            APP->StatusMessage((char *)"Please wait - Computing Gradient...");

            t1 = iftTic();

            this->gradient = iftImageBasins(this->iftorig, A);

            t2 = iftToc();

            APP->Data.arcw = APP->iftImageToScene(this->gradient);

            printf("Gradient Time: %.2f ms\n", iftCompTime(t1, t2));
            APP->StatusMessage((char *)"Done");
            APP->Unbusy();
        }
        else
        {
            this->gradient = APP->SceneToiftImage(APP->Data.arcw);

            printf("Gradient already computed... \n");
        }

        MaximumValue3(APP->Data.arcw);
        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);
    }

    void ModuleOIFT :: AllocBasicData()
    {
        if(this->iftorig != NULL) iftDestroyImage(&this->iftorig);

        this->iftorig = APP->SceneToiftImage(APP->Data.orig);

        if(this->fst != NULL) {
            iftDestroyAdjRel(&this->fst->A);
            iftDestroyImageForest(&this->fst);
        }

        if(this->igraph != NULL) iftDestroyIGraph(&this->igraph);
        if(this->Q != NULL) {
            iftFree(Q->value);
            iftDestroyDHeap(&this->Q);
        }
//
//        if(alloc_igraph) {
//        iftAdjRel *A = iftSpheric(1.0);
//        iftMImage *mimg = NULL;
//        iftImage *mask1 = NULL;
//            if (iftIsColorImage(this->iftorig)) {
//                mimg = iftImageToMImage(this->iftorig, LABNorm_CSPACE);
//            } else {
//                mimg = iftImageToMImage(this->iftorig, GRAY_CSPACE);
//            }
//
//            mask1 = iftSelectImageDomain(mimg->xsize, mimg->ysize, mimg->zsize);
//
//            /* minima of a basins manifold in that domain */
//            this->igraph = iftImplicitIGraph(mimg, mask1, A);
//
//            iftDestroyMImage(&mimg);
//            iftDestroyImage(&mask1);
//            iftDestroyAdjRel(&A);
//        }

        this->del_alg_type = ModuleOIFT::Watershed;

        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);
    }

    void ModuleOIFT :: DisplayParamPanel() {
        int x, y, w, h;

        OIFTParams *dialog = new OIFTParams(APP->Window, this);
        optDialog = (BaseDialog *)dialog;
        optDialog->ShowModal();
        APP->Window->GetPosition(&x, &y);
        APP->Window->GetSize(&w, &h);
        optDialog->Move(MAX(x - 20, 0), h / 2); //wxDefaultCoord);
    }


    void ModuleOIFT :: DisplayPanel() {
        int x, y, w, h;

        OIFTDialog *dialog = new OIFTDialog(APP->Window, this);
        optDialog = (BaseDialog *)dialog;
        optDialog->Show(true);
        APP->Window->GetPosition(&x, &y);
        APP->Window->GetSize(&w, &h);
        optDialog->Move(MAX(x - 20, 0), h / 2); //wxDefaultCoord);
    }


    iftSet *ModuleOIFT :: ProcessSeeds(iftLabeledSet **LS, int last_markerID)
    {
        Set *S = NULL, *Seeds = NULL;
        int p = 0, label = 0, id = 0, cod = 0, root_p = 0;
        iftSet *removal_marker = NULL;
        iftBMap *inSet = NULL;
        iftBMap *inSetRemoval =  NULL;

        if(this->igraph != NULL) {
            inSet = iftCreateBMap(this->igraph->index->n);
            inSetRemoval = iftCreateBMap(this->igraph->index->n);
        } else {
            inSet = iftCreateBMap(this->fst->img->n);
            inSetRemoval = iftCreateBMap(this->fst->img->n);
        }

        Seeds = APP->CopySeeds();

        *LS = NULL;

        S = Seeds;
        while (S != NULL)
        {
            p = S->elem;
            label = APP->GetSeedLabel(p); /* Seed label at pixel p*/
            id  = APP->GetSeedId(p); /* Marker id at pixel p*/
            cod = this->GetCodeValue(id, label); /* 'cod' is a code that contains the Marker ID (24 bits) + Label (8 bits) */

            if (APP->IsMarkedForRemoval(p))
            {
                if (!iftBMapValue(inSetRemoval, p))
                {
                    iftInsertSet(&removal_marker, p);
                    iftBMapSet1(inSetRemoval, p);
                }
            }
            else
            {
                if(last_markerID < id) {
                    iftInsertLabeledSetMarkerAndHandicap(LS, p, label, id, 0);
                }
            }

            // Updating the marker ID to match the one that was last added by the user
            this->markerID = iftMax(this->markerID, id);

            S = S->next;
        }

        DestroySet(&Seeds);
        iftDestroyBMap(&inSet);
        iftDestroyBMap(&inSetRemoval);
        return removal_marker;
    }


    void ModuleOIFT :: Run()
    {
        timer *t1, *t2;
        iftLabeledSet *LS = NULL, *aux;

        int tree_removal = 0, p = 0;
        iftSet *removal_markers = NULL;
        char msg[1024];

        APP->Busy(msg);
        APP->StatusMessage((char *)"Please wait - Computation in progress...");

        t1 = iftTic();

        switch (del_alg_type) {

            case ModuleOIFT::Watershed:
                removal_markers = this->ProcessSeeds(&LS, this->execution);

//                iftIGraphDiffWatershed(this->igraph, LS, removal_markers, this->Q);

                iftDiffWatershed(this->fst, LS, removal_markers);

                iftDestroySet(&removal_markers);
                iftDestroyLabeledSet(&LS);

                break;

            case ModuleOIFT::OIFTResuming:

                // Getting all markers selected from the beginning of times
                removal_markers = this->ProcessSeeds(&LS, IFT_NIL);

                // Performing segmentation from scratch
//                iftIGraphResetWatershed(igraph, Q);
                // NOTE: markers being removed can be disregarded
//                iftIGraphResumingDiffOrientedWatershed(igraph, this->presegmentation, LS, NULL, Q, this->ori_watershed_gamma);

                iftDiffOrientedWatershedResuming(this->fst, this->presegmentation, LS, removal_markers, this->ori_watershed_gamma);

                iftDestroySet(&removal_markers);
                iftDestroyLabeledSet(&LS);

                break;

            case ModuleOIFT::ISFResuming:
                // Getting all markers selected from the last execution
                removal_markers = this->ProcessSeeds(&LS, this->execution);

                iftIGraphDiffISF_Resuming_Root(this->igraph, this->presegmentation, LS, &removal_markers,
                                               this->Q, this->alpha, this->beta, this->gamma, true);

                iftDestroySet(&removal_markers);
                iftDestroyLabeledSet(&LS);

                break;
        };


        // Updating the last marker ID that was processed
        this->execution = this->markerID;

        t2 = iftToc();

        fprintf(stdout, "Diff. Watershed Time: %.2f ms\n", iftCompTime(t1, t2));


        /* This is required to display the label map */
        if(del_alg_type == ModuleOIFT::ISFResuming) {
            iftImage *tmp = iftIGraphLabel(this->igraph);
            iftWriteImageByExt(tmp, "label_%02d.scn", this->execution);
            iftDestroyImage(&tmp);

            APP->Data.label = APP->IGraphLabelImageToScene(this->igraph);
        } else {
            APP->Data.label = APP->LabelImageToScene(this->fst);
        }

        /* This removes the markers that were marked for removal */
        APP->DelMarkedForRemoval();

        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);

        APP->StatusMessage((char *)"Done");
        APP->Unbusy();

    }

    bool ModuleOIFT :: Stop()
    {
        static const char *title = {"Keep segmentation?"};
        char msg[IFT_STR_DEFAULT_SIZE];

        sprintf(msg, "You are about to leave the %s module.\nSave changes?", this->name);

        if (!this->active) return true;

        wxString wxtitle(title, wxConvUTF8);
        wxString wxmsg(msg, wxConvUTF8);

        wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);

        if (dialog.ShowModal() == wxID_YES)
            this->Finish();
        else
            this->FreeData();


        return true;
    }

    void ModuleOIFT :: FreeData()
    {

//APP->DetachOptPanel(optPanel);
        APP->SetDefaultInteractionHandler();
        APP->EnableObjWindow(true);
        APP->ResetData();
        APP->DrawSegmObjects();
        this->active = false;

        optDialog->Show(false);
        if (optDialog != NULL)
            optDialog->Destroy();
        optDialog = NULL;

        if(this->igraph != NULL) iftDestroyIGraph(&this->igraph);
        if(this->Q != NULL) {
            iftFree(Q->value);
            iftDestroyDHeap(&this->Q);
        }
        if(this->iftorig != NULL) iftDestroyImage(&this->iftorig);

        if(this->gradient != NULL) iftDestroyImage(&this->gradient);
        if(this->presegmentation != NULL) iftDestroyImage(&this->presegmentation);
        if(this->fst != NULL) {
            iftDestroyAdjRel(&this->fst->A);
            iftDestroyImageForest(&this->fst);
        }

    }


    void ModuleOIFT :: Finish()
    {
        int p, n, i;
        Scene *label = APP->Data.label;
        SegmObject *obj, *tmp;

        n = APP->Data.w * APP->Data.h * APP->Data.nframes;

// set visibility
        for (i = 0; i < APP->GetNumberOfObjs(); i++)
        {
            SegmObject *so;
            so = APP->GetObjByIndex(i);
            so->visibility = false;
        }

        for (i = 0; i < nobjs; i++)
        {
            obj = CreateSegmObject(this->obj_name[i], this->obj_color[i]);
            obj->mask = BMapNew(n);
            BMapFill(obj->mask, 0);
            for (p = 0; p < n; p++)
            {
                if (this->GetCodeLabel(label->data[p]) == i + 1)
                    _fast_BMapSet1(obj->mask, p);
            }
            obj->seed = APP->CopyMarkerList();
            APP->AddCustomObj(obj);
            APP->SetObjVisibility(obj->name, true);
        }

        this->FreeData();
    }


    void ModuleOIFT :: Reset()
    {
        static const char *title = "Reset segmentation?";
        static const char *msg = "Current segmentation will be lost.\nAre you sure you want to reset?";

        wxString wxtitle(title, wxConvUTF8);
        wxString wxmsg(msg, wxConvUTF8);

        wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);

        if (dialog.ShowModal() == wxID_YES)
        {
            iftIGraphResetWatershed(this->igraph, this->Q);

            APP->ResetData();
            this->markerID = IFT_NIL;


            APP->Refresh2DCanvas();
            APP->Refresh3DCanvas(true, 1.0);
        }
    }

    AdjRel *ModuleOIFT :: GetBrush()
    {
        OIFTDialog *iopt = (OIFTDialog *)optDialog;
        return iopt->GetBrush();
    }

    wxCursor *ModuleOIFT :: GetBrushCursor(int zoom)
    {
        OIFTDialog *iopt = (OIFTDialog *)optDialog;
        return iopt->GetBrushCursor(zoom);
    }

    void ModuleOIFT :: NextBrush()
    {
        OIFTDialog *iopt = (OIFTDialog *)optDialog;
        iopt->NextBrush();
    }

    void ModuleOIFT :: PrevBrush()
    {
        OIFTDialog *iopt = (OIFTDialog *)optDialog;
        iopt->PrevBrush();
    }

    int ModuleOIFT :: GetCodeID(int cod)
    {
        return (cod >> 8); //(int)label/256;
    }

    int ModuleOIFT :: GetCodeLabel(int cod)
    {
        return (cod & 0xff);  //(label % 256);
    }

    int ModuleOIFT :: GetCodeValue(int id, int lb)
    {
        return ((id << 8) | lb); //(id*256 + (int)lb);
    }
    void ModuleOIFT :: PrintSeedReport()
    {
        int p, l, Lmax = 0;
        int *hist = NULL;
        Set *S, *tmp;

        S = APP->CopySeeds();

        tmp = S;
        while (tmp != NULL)
        {
            p = tmp->elem;
            l = APP->GetSeedLabel(p);
            if (l > Lmax) Lmax = l;
            tmp = tmp->next;
        }

        hist = AllocIntArray(Lmax + 1);

        tmp = S;
        while (tmp != NULL)
        {
            p = tmp->elem;
            l = APP->GetSeedLabel(p);
            if (l >= 0 && l <= Lmax)
                hist[l]++;
            tmp = tmp->next;
        }
        printf("SeedReport:\n");
        for (l = 0; l <= Lmax; l++)
        {
            printf("\tLabel%02d: %d\n", l, hist[l]);
        }
        free(hist);
        DestroySet(&S);
    }

    void ModuleOIFT :: DeleteObj(int obj)
    {
        int o, p, n, cod, id, newcod, newlb;

        n = (APP->Data.label)->n;
        for (o = obj ; o < this->nobjs - 1; o++)
        {
            obj_color[o] = obj_color[o + 1];
            obj_visibility[o] = obj_visibility[o + 1];
            strcpy(obj_name[o], obj_name[o + 1]);
        }
        this->nobjs--;
        if (obj_sel > nobjs - 1) this->obj_sel--;

        for (p = 0; p < n; p++)
        {
            cod = (APP->Data.label)->data[p];
            id = GetCodeID(cod);
            if (GetCodeLabel(cod) == obj + 1)
            {
                newcod = GetCodeValue(id, 0);
                (APP->Data.label)->data[p] = newcod;
                if (APP->IsSeed(p))
                    APP->AddSeed(p, 0, id);
                APP->SetLabelColour(newcod, NIL);
            }
            else if (GetCodeLabel(cod) > obj + 1)
            {
                newlb  = GetCodeLabel(cod) - 1;
                newcod = GetCodeValue(id, newlb);
                (APP->Data.label)->data[p] = newcod;
                if (APP->IsSeed(p))
                    APP->AddSeed(p, newlb, id);

                if (obj_visibility[newlb - 1])
                    APP->SetLabelColour(newcod, obj_color[newlb - 1]);
                else
                    APP->SetLabelColour(newcod, NIL);
            }
        }
        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);
    }

} //end OIFT namespace





