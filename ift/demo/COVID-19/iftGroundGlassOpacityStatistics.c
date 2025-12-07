//
// Created by azaelmsousa on 20/08/20.
//

#include "ift.h"

int main(int argc, const char *argv[])
{

    if (argc != 5)
        iftError("Usage: iftGroundGlassOpacityStatistics ...\n[1] Input original image\n[2] Lungs Segmentation image\n[3] GGO segmentation\n[4] Output dir","main");

    iftImage *segm = iftReadImageByExt(argv[2]);
    iftImage *ggo = iftReadImageByExt(argv[3]);
    int k_top = 5;

    if (!iftIsDomainEqual(segm,ggo))
        iftError("Both masks need to be on the same domain.","main");

    iftAdjRel *A = iftSpheric(sqrt(3.));
    iftImage *comp = iftLabelComp(ggo,A);
    iftDestroyAdjRel(&A);

    // Computing total number of lesions and lesions on each lung
    iftLabeledSet *L = iftGeometricCenters(comp);
    int n_lesions_ll = 0;
    int n_lesions_rl = 0;
    while (L != NULL){
        int label = 0;
        int elem = iftRemoveLabeledSet(&L,&label);
        if (segm->val[elem] == 1)
            n_lesions_rl++;
        else if (segm->val[elem] == 2)
            n_lesions_ll++;
    }
    int n_lesions = n_lesions_ll+n_lesions_rl;

    // Computing number of spels compromised by GGO of each lung
    int sum_ggo_rl = 0;
    int sum_ggo_ll = 0;
    for (int i = 0; i < comp->n; i++) {
        if (comp->val[i] > 0) {
            if (segm->val[i] == 1)
                sum_ggo_rl++;
            else if (segm->val[i] == 2)
                sum_ggo_ll++;
        }
    }

    // Computing number of spels of each lung
    int sum_rl = 0;
    int sum_ll = 0;
    for (int i = 0; i < segm->n; i++){
        if (segm->val[i] == 1)
            sum_rl++;
        else if (segm->val[i] == 2)
            sum_ll++;
    }

    // Getting top K largest lesions
    iftImage *orig = iftReadImageByExt(argv[1]);
    A = iftSpheric(sqrt(3.));
    iftImage *largest = NULL;
    if (n_lesions < k_top){
        largest = iftCopyImage(comp);
    } else {
        largest = iftSelectKLargestComp(comp, A, k_top);
        iftImage *relabel = iftRelabelImage(largest);
        iftDestroyImage(&largest);
        largest = relabel;
        relabel = NULL;
    }
    iftDestroyImage(&comp);
    L = iftGeometricCenters(largest);
    iftColorTable *ctb = iftCategoricalColorTable(iftMaximumValue(largest)+1);

    while (L != NULL){
        int label;
        int elem = iftRemoveLabeledSet(&L,&label);
        iftVoxel u = iftGetVoxelCoord(largest,elem);

        iftImage *slice = iftGetXYSlice(orig,u.z);
        iftImage *norm = iftNormalize(slice,0,255);
        iftImage *largest_obj = iftExtractObject(largest,largest->val[elem]);
        iftImage *slice_label = iftGetXYSlice(largest_obj,u.z);
        printf("obj %d: %.2f\n",label,iftAreaVolumeOfObject(largest_obj,label));

        iftDrawLabels(norm,slice_label,ctb,A,TRUE,0.5);

        char filename[200];
        sprintf(filename,"%s/%s_%d.png",argv[4],iftFilename(argv[1],iftFileExt(argv[1])),u.z);
        iftWriteImageByExt(norm,filename);

        iftDestroyImage(&slice);
        iftDestroyImage(&slice_label);
        iftDestroyImage(&norm);
        iftDestroyImage(&largest_obj);
    }
    iftDestroyColorTable(&ctb);
    iftDestroyAdjRel(&A);
    iftDestroyImage(&largest);
    iftDestroyImage(&orig);


    FILE *f;
    char csvname[200];
    bool firstline = FALSE;
    sprintf(csvname,"%s/statistics_%s.csv",argv[4],iftFilename(argv[1],iftFileExt(argv[1])));
    if (!iftFileExists(csvname))
        firstline = TRUE;
    f = fopen(csvname,"a+");
    if (f == NULL)
        iftError("Cannot open csv file","main");
    if (firstline)
        fprintf(f,"patient;%% total ggo;%% right lung ggo;%% left lung ggo\n");
    fprintf(f,"%s;%f;%f;%f\n",
            iftFilename(argv[1],NULL),                      //patient
            ((float)sum_ggo_ll+sum_ggo_rl)/(sum_ll+sum_rl), //total ggo
            ((float)sum_ggo_rl)/sum_rl,                     //right lung ggo
            ((float)sum_ggo_ll)/sum_ll);                    //left lung ggo
    fclose(f);

    return 0;
}