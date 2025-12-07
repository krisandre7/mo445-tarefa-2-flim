#include "moduledrift.h"
#include "driftdialog.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace DRIFT
{

    ModuleDRIFT :: ModuleDRIFT()
            : SegmentationModule()
    {
      SetName((char *)"Diff. Relaxed IFT");
      SetAuthor((char *)"Nikolas Moya");
      SoftwareVersion ver(2, 0, 0);
      SetVersion(ver);

      this->InitNull();
      cost = NULL;
      pred = NULL;

      optDialog = NULL;

      obj_visibility[0] = true;
      obj_color[0] = 0xffff00;
      obj_name[0][0] = '\0';
      obj_sel = 0;
      nobjs = 0;
    }


    ModuleDRIFT :: ~ModuleDRIFT()
    {

    }

    void ModuleDRIFT :: Start()
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

      this->markerID = 4;
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
        }
      }

      this->ComputeGradientImageBasins();

      if (!resume)
      {
        APP->ResetData();
        r = APP->ShowNewObjDialog(&this->obj_color[0], this->obj_name[0]);
        if (r != 0) return;

        // /* This will be removed soon */
        // while (strcasecmp(this->obj_name[0], "All") == 0)
        // {
        //   wxMessageDialog *dialog = new wxMessageDialog(APP->Window, _T("Resuming All objects is not working."), _T("Resume All"), wxOK, wxDefaultPosition);
        //   dialog->ShowModal();
        //   r = APP->ShowNewObjDialog(&this->obj_color[0], this->obj_name[0]);
        //   if (r != 0) return;
        // }
        this->active = true;
        this->nobjs = 1;
        this->obj_sel = 0;
        APP->SetLabelColour(0, NIL);
        APP->SetLabelColour(1, this->obj_color[0]);

        this->AllocData();
      }
      else
      {
        this->active = true;
        this->nobjs = number_of_objects;
        this->obj_sel = 0;
        APP->SetLabelColour(0, NIL);
        if (APP->GetNumberOfObjs() == number_of_objects)
        {
          SegmObject *aux = NULL;
          for (int i = 0; i < number_of_objects; i++)
          {
            aux = APP->GetObjByIndex(i);
            strcpy(this->obj_name[i], aux->name);
            this->obj_color[i] = aux->color;
            this->obj_visibility[i] = true;
          }
        }

        this->AllocData();
        //----- resuming previous segmentation: ------------
        APP->Busy((char *)"Please wait, resuming previous segmentation...");
        APP->StatusMessage((char *)"Please wait - Resuming Segmentation...");

        iftDestroyImageForest(&this->forest);
        iftImage *label_image = APP->SceneToiftImage(APP->Data.label);
        iftAdjRel *A = iftSpheric(1.0);
        iftLabeledSet *segm_seeds = NULL, *tmp = NULL;
        this->forest = iftCreateImageForest(this->gradient, A);
        this->markerID = 0;
        tmp = iftLabelToForestGeodesicRobot(this->gradient, label_image, 50, iftMaximumValue(label_image), 1, 5, 1);
        while (tmp != NULL)
        {
          iftInsertLabeledSetMarkerAndHandicap(&segm_seeds, tmp->elem, tmp->label, this->markerID++, 0);
          tmp = tmp->next;
        }
        iftDiffWatershed(this->forest, segm_seeds, NULL);

        iftDblArray *error = iftDiceSimilarityMultiLabel(label_image, forest->label, number_of_objects);
        printf("Dice: %f\n", error->val[0]);
        printf("Number of Markers: %d\n", this->markerID);
        iftDestroyDblArray(&error);


        APP->Data.label = APP->LabelImageToScene(this->forest);

        int p, label, id, cod;
        for(p=0; p<this->forest->pred->n; p++)
        {
          if (this->forest->pred->val[p] == NIL)
          {
            label = APP->GetSeedLabel(p);
            id = APP->GetSeedId(p);
            cod = this->GetCodeValue(id, label);
            if (label == 0)
              APP->SetLabelColour(cod, NIL);
            else
              APP->SetLabelColour(cod, this->obj_color[label-1]);
          }
        }

        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);

        APP->StatusMessage((char *)"Done");
        APP->Unbusy();
      }

      // // ----- The object already exists: ------------
      // wxString wxtitle(title, wxConvUTF8);
      // wxString wxmsg(msg, wxConvUTF8);

      // wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);

      // if (dialog.ShowModal() != wxID_YES) return;
      // //----- resuming previous segmentation: ------------
      // APP->Busy((char *)"Please wait, resuming previous segmentation...");
      // APP->StatusMessage((char *)"Please wait - Resuming Segmentation...");
      // StartTimer(&tic);

      // printf("Here\n");

      // bin = CreateScene(APP->Data.w, APP->Data.h, APP->Data.nframes);
      // CopySegmObjectMask2Scene(obj, bin);

      // this->SetWf(APP->Data.arcw, bin);

      // ResumeFromScratchDIFT_MinSeeds(APP->Data.orig, this->Wf, bin, cost, pred, APP->Data.label, &S);
      // for (p = 0; p < bin->n; p++)
      // {
      //     id = (APP->Data.label)->data[p] + 4;
      //     cod = this->GetCodeValue(id, bin->data[p]);
      //     (APP->Data.label)->data[p] = cod;
      // }
      // this->markerID = 0;
      // aux = S;
      // while (aux != NULL)
      // {
      //     p  = aux->elem;
      //     cod = (APP->Data.label)->data[p];
      //     id = this->GetCodeID(cod);

      //     if (bin->data[p] > 0) APP->SetLabelColour(cod, this->obj_color[0]);
      //     else               APP->SetLabelColour(cod, NIL);

      //     APP->AddSeed(p, bin->data[p], id);
      //     this->markerID = MAX(this->markerID, id);
      //     aux = aux->next;
      // }
      // this->markerID++;
      // DestroyScene(&bin);

      // printf("\nResuming previous segmentation - Time: ");
      // StopTimer(&tic);

      // //this->PrintSeedReport();

      // APP->StatusMessage((char *)"Done");
      // APP->Unbusy();

      // APP->Refresh2DCanvas();
      // APP->Refresh3DCanvas(true, 1.0);

      //----- The object already exists: ------------
      // wxString wxtitle(title, wxConvUTF8);
      // wxString wxmsg(msg, wxConvUTF8);

      // wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);

      // if (dialog.ShowModal() != wxID_YES) return;
      // //----- resuming previous segmentation: ------------
      // APP->Busy((char *)"Please wait, resuming previous segmentation...");
      // APP->StatusMessage((char *)"Please wait - Resuming Segmentation...");
      // StartTimer(&tic);

      // bin = CreateScene(APP->Data.w, APP->Data.h, APP->Data.nframes);
      // CopySegmObjectMask2Scene(obj, bin);

      // this->SetWf(APP->Data.arcw, bin);

      // ResumeFromScratchDIFT_MinSeeds(APP->Data.orig, this->Wf, bin, cost, pred, APP->Data.label, &S);
      // for (p = 0; p < bin->n; p++)
      // {
      //     id = (APP->Data.label)->data[p] + 4;
      //     cod = this->GetCodeValue(id, bin->data[p]);
      //     (APP->Data.label)->data[p] = cod;
      // }
      // this->markerID = 0;
      // aux = S;
      // while (aux != NULL)
      // {
      //     p  = aux->elem;
      //     cod = (APP->Data.label)->data[p];
      //     id = this->GetCodeID(cod);

      //     if (bin->data[p] > 0) APP->SetLabelColour(cod, this->obj_color[0]);
      //     else               APP->SetLabelColour(cod, NIL);

      //     APP->AddSeed(p, bin->data[p], id);
      //     this->markerID = MAX(this->markerID, id);
      //     aux = aux->next;
      // }
      // this->markerID++;
      // DestroyScene(&bin);

      // printf("\nResuming previous segmentation - Time: ");
      // StopTimer(&tic);

      // //this->PrintSeedReport();

      // APP->StatusMessage((char *)"Done");
      // APP->Unbusy();

      // APP->Refresh2DCanvas();
      // APP->Refresh3DCanvas(true, 1.0);
    }

    void ModuleDRIFT :: InitNull()
    {
      this->iftorig = NULL;
      this->gradient = NULL;

      this->seedimage = NULL;
      this->forest = NULL;

      this->smooth = NULL;
      this->gui_markers = NULL;

      this->largest_id = 9999;
      this->execution = 0;
    }

    void ModuleDRIFT :: ComputeGradientImageBasins()
    {
      iftAdjRel *adj = iftSpheric(1.0);
      timer *t1, *t2;

      if (MaximumValue3(APP->Data.arcw) == 0)
        DestroyScene(&(APP->Data.arcw));

      if (this->iftorig != NULL) iftDestroyImage(&this->iftorig);
      if (this->gradient != NULL) iftDestroyImage(&this->gradient);

      this->iftorig = APP->SceneToiftImage(APP->Data.orig);

      if (APP->Data.arcw == NULL)
      {
        APP->Busy((char *)"Please wait, computing gradient...");
        APP->StatusMessage((char *)"Please wait - Computing Gradient...");

        t1 = iftTic();
        this->gradient = iftImageBasins(this->iftorig, adj);
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

      iftDestroyAdjRel(&adj);

      MaximumValue3(APP->Data.arcw);
      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
    }

    void ModuleDRIFT :: AllocData()
    {
      iftAdjRel *adj = iftSpheric(1.0);
      int x, y, w, h;
      //DIFT data:
      if (pred != NULL) DestroyScene(&pred);
      if (cost != NULL) bia::Scene16::Destroy(&cost);
      if (Wf  != NULL) bia::Scene16::Destroy(&Wf);

      pred = CreateScene(APP->Data.w, APP->Data.h, APP->Data.nframes);
      cost = bia::Scene16::Create(APP->Data.w, APP->Data.h, APP->Data.nframes);
      Wf   = bia::Scene16::Create(APP->Data.w, APP->Data.h, APP->Data.nframes);

      if (this->forest != NULL) iftDestroyImageForest(&this->forest);
      this->forest = iftCreateImageForest(this->gradient, adj);
      this->smooth = iftCreateSmoothBorder(this->gradient, this->forest->A, 5, 0.5);
      this->seedimage = iftCreateImage(this->iftorig->xsize, this->iftorig->ysize, this->iftorig->zsize);
      this->gui_markers = iftCreateImage(this->iftorig->xsize, this->iftorig->ysize, this->iftorig->zsize);
      iftSetImage(this->seedimage, -1);
      iftSetImage(this->gui_markers, 0);

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);

      DRIFTDialog *dialog = new DRIFTDialog(APP->Window, this);
      optDialog = (BaseDialog *)dialog;
      optDialog->Show(true);
      APP->Window->GetPosition(&x, &y);
      APP->Window->GetSize(&w, &h);
      optDialog->Move(MAX(x - 20, 0), h / 2); //wxDefaultCoord);
    }



    iftSet *ModuleDRIFT :: ProcessSeeds(iftLabeledSet **LS)
    {
      Set *S = NULL, *Seeds = NULL;
      int p = 0, label = 0, id = 0, cod = 0, root_p = 0;
      iftSet *removal_marker = NULL;
      iftBMap *inSet = iftCreateBMap(this->forest->img->n);
      iftBMap *inSetRemoval = iftCreateBMap(this->forest->img->n);

      Seeds = APP->CopySeeds();

      S = Seeds;
      while (S != NULL)
      {
        p = S->elem;
        label = APP->GetSeedLabel(p); /* Seed label at pixel p*/
        id = APP->GetSeedId(p); /* Marker id at pixel p*/
        cod = this->GetCodeValue(id, label); /* 'cod' is a code that contains the Marker ID (24 bits) + Label (8 bits) */

        if (APP->IsMarkedForRemoval(p))
        {
          this->gui_markers->val[p] = 0;
          if (!iftBMapValue(inSetRemoval, p))
          {
            iftInsertSet(&removal_marker, p);
            iftBMapSet1(inSetRemoval, p);
          }
        }
        else
        {
          if (this->gui_markers->val[p] == 0 && this->execution > 0)
          {
            root_p = this->forest->root->val[p];
            //if (this->gui_markers->val[root_p] < 0)
            if (this->forest->pathval->val[root_p] != 0 && this->gui_markers->val[root_p] == 0)
            {
              if (!iftBMapValue(inSetRemoval, root_p) && root_p != p)
              {
                this->gui_markers->val[root_p] = 0;
                iftInsertSet(&removal_marker, root_p);
              }
            }
          }
          if (this->seedimage->val[p] != label)
          {
            this->gui_markers->val[p] = cod;
            if (!iftBMapValue(inSet, p))
            {
              iftInsertLabeledSetMarkerAndHandicap(LS, p, label, id, 0);
              iftBMapSet1(inSet, p);
            }
          }
        }
        S = S->next;
      }
      // iftImage *copy = iftGetXYSlice(this->gui_markers, 58);
      // for (int p = 0; p < copy->n; p++)
      // {
      //   if (copy->val[p] < 0)
      //     copy->val[p] = copy->val[p] * -1;
      //   if (copy->val[p] != 0)
      //     copy->val[p] = (copy->val[p] & 0xff) + 1;
      // }
      // iftNormalize(copy, 0, 255);
      // char buffer[128];
      // sprintf(buffer, "gui_markers%d.pgm", this->execution);
      // iftWriteImageP2(iftGetXYSlice(this->iftorig, 58), "ZX Slice.pgm");
      // iftWriteImageP2(copy, buffer);
      // iftDestroyImage(&copy);


      DestroySet(&Seeds);
      iftDestroyBMap(&inSet);
      iftDestroyBMap(&inSetRemoval);
      return removal_marker;
    }

// iftFImage **ModuleDRIFT :: pvtCreateObjectEDTFromGT(int number_of_objects, iftImage *gt_image)
// {
//   int p, o;
//   iftImage *temp_image;
//   iftAdjRel *sphericsqrt3 = iftSpheric(sqrtf(3.0));
//   iftFImage **object_edt_gt = (iftFImage **) malloc(sizeof(iftFImage *) * (number_of_objects + 2));

//   object_edt_gt[0] = NULL;
//   printf("Generating individual GTs for ASD... ");
//   for (o = 1; o <= number_of_objects; o++)
//   {
//     temp_image = iftCreateImage(gt_image->xsize, gt_image->ysize, gt_image->zsize);
//     for (p = 0; p < gt_image->n; p++)
//     {
//       if (gt_image->val[p] == o)
//         temp_image->val[p] = 1;
//     }
//     object_edt_gt[o] = iftSignedDistTrans(temp_image, sphericsqrt3);
//     iftDestroyImage(&temp_image);
//   }
//   iftDestroyAdjRel(&sphericsqrt3);
//   printf("[DONE]\n");
//   return object_edt_gt;
// }
    void ModuleDRIFT :: Run()
    {
      timer *t1, *t2;
      iftLabeledSet *LS = NULL, *aux;
      int smooth_iterations = spin->GetValue();
      int tree_removal = 0, p = 0;
      iftSet *removal_markers = NULL;
      char msg[1024];

      if (checksmooth->IsChecked())
        sprintf(msg, "Please wait, DRIFT and smoothing are running... ");
      else
        sprintf(msg, "Please wait, only Diff. Relaxed IFT is running... ");
      APP->Busy(msg);
      APP->StatusMessage((char *)"Please wait - Computation in progress...");

      removal_markers = this->ProcessSeeds(&LS);

      t1 = iftTic();
      iftDiffWatershed(this->forest, LS, removal_markers);

      this->execution++;
      t2 = iftToc();
      fprintf(stdout, "Diff. Watershed Time: %.2f ms\n", iftCompTime(t1, t2));

      if (checksmooth->IsChecked() && smooth_iterations >= 1 )//&& tree_removal == 0)
      {
        t1 = iftTic();
        this->smooth->smooth_iterations = smooth_iterations;
        iftRelaxObjects(this->forest, this->smooth);
        t2 = iftToc();
        fprintf(stdout, "Smoothing time: %.2f ms\n", iftCompTime(t1, t2));
      }
      /* This is required to display the label map */
      APP->Data.label = APP->LabelImageToScene(this->forest);

      this->largest_id = -1;
      for (p = 0; p < this->seedimage->n; p++)
      {
        if (APP->Data.marker != NULL)
        if (APP->Data.marker->data[p] > this->largest_id)
          this->largest_id = APP->Data.marker->data[p];

        if (this->forest->pred->val[p] == NIL)
        {
          this->seedimage->val[p] = this->forest->label->val[p];
          if (this->gui_markers->val[p] == 0)
            this->gui_markers->val[p] = -GetCodeValue(this->forest->marker->val[p], this->forest->label->val[p]);
          else
            this->gui_markers->val[p] = GetCodeValue(this->forest->marker->val[p], this->forest->label->val[p]);
        }
        else
          this->seedimage->val[p] = NIL;
      }


      //printf("Is Segmentation Consistent? %d\n", iftIsSegmentationConsistent(this->forest));


      // printf("Computing error from current segmentation to base/test_brain/labels/000001_0000012.scn\n");

      // iftFImage **object_edt_gt = NULL;
      // iftImage *gt_image = NULL;
      // gt_image = iftReadImage("/home/nmoya/Documents/base/test_brain/labels/000001_0000012.scn");
      // object_edt_gt = pvtCreateObjectEDTFromGT(iftMaximumValue(gt_image), gt_image);

      // float *asd_array = NULL;

      // asd_array = iftASDError(forest->label, object_edt_gt, gt_image->maxval);
      // // Compute the error in (mm)
      // for (int o = 0; o <= gt_image->maxval; o++){
      //   asd_array[o] = (asd_array[o] * gt_image->dx);
      //   printf("%d: %.2f\n", o, asd_array[o]);
      // }

      // free(asd_array);


      // iftImage *gt_image = NULL;
      // gt_image = iftReadImage("/home/nmoya/Documents/base/test_brain/labels/000001_0000012.scn");

      // float *asd_array = NULL;

      // asd_array = iftDiceSimilarityMultiLabel(forest->label, gt_image, gt_image->maxval);
      // // Compute the error in (mm)
      // for (int o = 0; o <= gt_image->maxval; o++)
      //   printf("%d: %.2f\n", o, asd_array[o]);

      // free(asd_array);


      // if (object_edt_gt != NULL)
      // {
      //   for (int o = 1; o <= gt_image->maxval; o++)
      //     iftDestroyFImage(&object_edt_gt[o]);
      //   free(object_edt_gt);
      //   object_edt_gt = NULL;
      // }
      // iftDestroyImage(&gt_image);

      // printf("-----------------------------------\n");




      // This removes the markers that were marked for removal
      APP->DelMarkedForRemoval();

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);

      APP->StatusMessage((char *)"Done");
      APP->Unbusy();

    }

    void ModuleDRIFT :: Relax()
    {
      timer *t1, *t2;
      char buffer[1024];
      int p, q, i;
      int smooth_iterations = spin->GetValue();

      // wxMessageDialog *dialog = new wxMessageDialog(APP->Window, _T("This module is under maintance. Come back later. "), _T("Relax boundary"), wxOK, wxDefaultPosition);
      // dialog->ShowModal();
      // return;

      sprintf(buffer, "Please wait, global boundary smoothing of %d iterations...", smooth_iterations);
      APP->Busy(buffer);
      APP->StatusMessage((char *)"Please wait - Smoothing the image...");

      if (smooth_iterations >= 1)
      {
        // This fills the processed map as 1 to compute the frontier of the whole image
        iftFillBMap(this->forest->processed, 1);

        t1 = iftTic();
        this->smooth->smooth_iterations = smooth_iterations;
        iftRelaxObjects(this->forest, this->smooth);
        t2 = iftToc();
        fprintf(stdout, "Consistent Smoothing (%d it.): %.2f ms\n", smooth_iterations, iftCompTime(t1, t2));
        APP->Data.label = APP->LabelImageToScene(this->forest);

        // printf("Is Segmentation Consistent? %d\n", iftIsSegmentationConsistent(this->forest));

      }

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);

      APP->StatusMessage((char *)"Done");
      APP->Unbusy();
    }
    bool ModuleDRIFT :: Stop()
    {
      static const char *title = {"Keep segmentation?"};
      static const char *msg = { "You are about to leave the Diff. Relaxed IFT module.\nSave changes?" };

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

    void ModuleDRIFT :: FreeData()
    {
      //DIFT data:
      if (pred != NULL) DestroyScene(&pred);
      if (cost != NULL) bia::Scene16::Destroy(&cost);
      if (Wf  != NULL) bia::Scene16::Destroy(&Wf);

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

      if (this->forest != NULL) iftDestroyImageForest(&this->forest);
      if (this->iftorig != NULL) iftDestroyImage(&this->iftorig);
      if (this->seedimage != NULL) iftDestroyImage(&this->seedimage);
      if (this->gui_markers != NULL) iftDestroyImage(&this->gui_markers);
      if (this->gradient != NULL) iftDestroyImage(&this->gradient);
      if (this->smooth != NULL) iftDestroySmoothBorder(&this->smooth);
    }


    void ModuleDRIFT :: Finish()
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


    void ModuleDRIFT :: Reset()
    {
      static const char *title = "Reset segmentation?";
      static const char *msg = "Current segmentation will be lost.\nAre you sure you want to reset?";

      wxString wxtitle(title, wxConvUTF8);
      wxString wxmsg(msg, wxConvUTF8);

      wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);

      if (dialog.ShowModal() == wxID_YES)
      {

        bia::Scene16::Fill(this->cost, USHRT_MAX);
        SetScene(this->pred, NIL);

        iftResetImageForest(this->forest);
        iftSetImage(this->seedimage, -1);

        APP->ResetData();
        this->markerID = 4;


        APP->Refresh2DCanvas();
        APP->Refresh3DCanvas(true, 1.0);
      }
    }



// tmp = delSet;
// while (tmp != NULL) /* Add as a seed with label -1 all pixels within the marker ID area marked for removal */
// {
//     if (tmp->elem == id)
//     {
//         tree_removal = 1;
//         this->gui_markers->val[p] = NIL;
//         iftInsertLabeledSetMarkerAndHandicap(LSRemoval, p, NIL, id, this->forest->img->val[p]);
//     }
//     tmp = tmp->next;
// }




// for (p = 0; p < this->seedimage->n; p++)
// {
//     if (this->prev_seedimage->val[p] == this->seedimage->val[p])
//         this->prev_seedimage->val[p] = NIL;
//     else
//         this->prev_seedimage->val[p] = this->seedimage->val[p];
// }
// for (p = 0; p < this->prev_seedimage->n; p++)
// {
//     label = this->prev_seedimage->val[p];
//     if(label != NIL)
//         iftInsertLabeledSetMarkerAndHandicap(LS, p, label, markers->val[p], this->forest->img->val[p]);
// }




    AdjRel *ModuleDRIFT :: GetBrush()
    {
      DRIFTDialog *iopt = (DRIFTDialog *)optDialog;
      return iopt->GetBrush();
    }

    wxCursor *ModuleDRIFT :: GetBrushCursor(int zoom)
    {
      DRIFTDialog *iopt = (DRIFTDialog *)optDialog;
      return iopt->GetBrushCursor(zoom);
    }

    void ModuleDRIFT :: NextBrush()
    {
      DRIFTDialog *iopt = (DRIFTDialog *)optDialog;
      iopt->NextBrush();
    }

    void ModuleDRIFT :: PrevBrush()
    {
      DRIFTDialog *iopt = (DRIFTDialog *)optDialog;
      iopt->PrevBrush();
    }

    int ModuleDRIFT :: GetCodeID(int cod)
    {
      return (cod >> 8); //(int)label/256;
    }

    int ModuleDRIFT :: GetCodeLabel(int cod)
    {
      return (cod & 0xff);  //(label % 256);
    }

    int ModuleDRIFT :: GetCodeValue(int id, int lb)
    {
      return ((id << 8) | lb); //(id*256 + (int)lb);
    }
    void ModuleDRIFT :: PrintSeedReport()
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
    void ModuleDRIFT :: MarkForRemovalIsolatedSeeds()
    {
      Scene *scn = APP->Data.orig;
      int i, p, q, l;
      Voxel u, v;
      AdjRel3 *A = Spheric(1.0);
      Set *S = APP->CopySeeds();
      Set *tmp = NULL;
      bool alldiff;

      tmp = S;
      while (tmp != NULL)
      {
        p = tmp->elem;
        l = APP->GetSeedLabel(p);

        u.x = VoxelX(scn, p);
        u.y = VoxelY(scn, p);
        u.z = VoxelZ(scn, p);
        alldiff = true;
        for (i = 1; i < A->n; i++)
        {
          v.x = u.x + A->dx[i];
          v.y = u.y + A->dy[i];
          v.z = u.z + A->dz[i];
          if (ValidVoxel(scn, v.x, v.y, v.z))
          {
            q = v.x + scn->tby[v.y] + scn->tbz[v.z];
            if (l == this->GetCodeLabel(APP->Data.label->data[q]))
              alldiff = false;
          }
        }
        if (alldiff)
        {
          APP->MarkForRemoval(APP->GetSeedId(p));
        }
        tmp = tmp->next;
      }
      DestroySet(&S);
      DestroyAdjRel3(&A);
    }
    void ModuleDRIFT :: DeleteObj(int obj)
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
    void ModuleDRIFT :: Run_DIFT()
    {
      Set *S = NULL;
      Set *delSet = NULL, *aux;
      int id, cod;

      S = APP->CopySeeds();
      delSet = APP->CopyIdsMarkedForRemoval();
      aux = delSet;
      while (aux != NULL)
      {
        id = aux->elem;
        cod = GetCodeValue(id, APP->GetSeedLabelById(id));
        aux->elem = cod;
        aux = aux->next;
      }

      RunDIFT(this->Wf, cost,
              pred, APP->Data.label,
              &S, &delSet);
      APP->DelMarkedForRemoval();
      DestroySet(&S);
      DestroySet(&delSet);
    }
    void ModuleDRIFT :: SetWf(Scene *Wl, Scene *bin)
    {
      int Imax = MaximumValue3(Wl);
      int p, q, i;
      bia::Voxel u, v;
      bia::AdjRel3::AdjRel3 *A = bia::AdjRel3::Spheric(1.0);
      float a, w_c, w_l;

      if (bin == NULL) a = 0.0;
      else          a = 0.4;

      for (u.c.z = 0; u.c.z < Wl->zsize; u.c.z++)
      {
        for (u.c.y = 0; u.c.y < Wl->ysize; u.c.y++)
        {
          for (u.c.x = 0; u.c.x < Wl->xsize; u.c.x++)
          {
            p = VoxelAddress(Wl, u.c.x, u.c.y, u.c.z);
            w_c = 0.0;
            if (bin != NULL)
            {
              for (i = 1; i < A->n; i++)
              {
                v.v = u.v + A->d[i].v;
                if (!ValidVoxel(Wl, v.c.x, v.c.y, v.c.z)) continue;
                q = VoxelAddress(Wl, v.c.x, v.c.y, v.c.z);
                if (bin->data[p] != bin->data[q])
                {
                  w_c = 1.0;
                  break;
                }
              }
            }
            if (Imax == 0) w_l = 0.0;
            else        w_l = ((float)Wl->data[p]) / ((float)Imax);

            (this->Wf)->data[p] = ROUND(65000 * ((1.0 - a) * w_l + a * w_c));
          }
        }
      }
      bia::Scene16::GetMaximumValue(Wf);
      bia::AdjRel3::Destroy(&A);
    }

} //end DRIFT namespace





