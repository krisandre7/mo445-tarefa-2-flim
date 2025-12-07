//
// Created by azaelmsousa on 20/09/21.
//

#include "ift.h"

int main(int argc, char *argv[])
{
    if (argc != 5){
        iftError("Usage: iftISBI_metrics <original ALTIS segmentation> <ALTIS correction> <Anomaly masks> <output file .csv>","main");
    }

    iftImage *orig_altis = iftReadImageByExt(argv[1]);
    iftImage *corr_altis = iftReadImageByExt(argv[2]);
    iftImage *anomalies  = iftReadImageByExt(argv[3]);

    /* Metric 01 - The agreement level of the original ALTIS and the corrected one.
     * The idea is to measure the impact of the correction in the original segmentation
     * It is computed with the following algorithm:
     * Input <= Let A be the ALTIS original segmentation, B the correction and G the anomalies GT.
     * (1) Compute the residue image R=B-A.
     * (2) Remove voxels of R that belong to G as |R\G|.
     * (3) Compute the relative error e = |R\G|/|B|.
     * (4) The agreement level is given as p = 1 - e.
       -----------------------*/
    // Step 1
    iftImage *R = iftSubReLU(corr_altis,orig_altis);
    // Step 2
    for (int i = 0; i < R->n; i++)
    {
        if ((R->val[i] > 0) && (anomalies->val[i] > 0))
            R->val[i] = 0;
    }
    // Step 3
    float e;
    iftImage *R_th      = iftThreshold(R,1,iftMaximumValue(R),1);
    iftImage *B_th      = iftThreshold(corr_altis,1,iftMaximumValue(corr_altis),1);
    int R_spels         = iftCountObjectSpels(R_th,1);
    int B_spels         = iftCountObjectSpels(B_th,1);
    e = ((float)R_spels)/B_spels;
    // Step 4
    float p = 1 - e;

    // Free memory
    iftDestroyImage(&R);
    iftDestroyImage(&R_th);
    iftDestroyImage(&B_th);

    /* Metric 02 - Percentage of anomaly voxels that was missed by the segmentation
     * The idea is to compute the amount of anomaly spels that belongs to the
     * segmentation both the original and the corrected.
     * It is computed with the following algorihtm:
     * Input <= Let A be the ALTIS original segmentation, B the correction and G the anomalies GT.
     * (1) Remove voxels of G that belong to M in {A,B} as |G\M|.
     * (2) Compute |G\M|/|G|.
       -----------------------*/
    // Step 1
    iftImage *G1 = iftCreateImageFromImage(anomalies);
    iftImage *G2 = iftCreateImageFromImage(anomalies);
    // --- with M as A and B
    for (int i = 0; i < anomalies->n; i++){
        if ((anomalies->val[i] > 0) && (orig_altis->val[i] == 0))
            G1->val[i] = 1;
        if ((anomalies->val[i] > 0) && (corr_altis->val[i] == 0))
            G2->val[i] = 1;
    }
    // Step 2
    iftImage *G1_th        = iftThreshold(G1,1,iftMaximumValue(G1),1);
    iftImage *G2_th        = iftThreshold(G2,1,iftMaximumValue(G2),1);
    iftImage *anomalies_th = iftThreshold(anomalies,1,iftMaximumValue(anomalies),1);
    int G1_spels           = iftCountObjectSpels(G1_th,1);
    int G2_spels           = iftCountObjectSpels(G2_th,1);
    int anomalies_spels    = iftCountObjectSpels(anomalies_th,1);

    float perc_G1 = ((float)G1_spels)/anomalies_spels;
    float perc_G2 = ((float)G2_spels)/anomalies_spels;

    // Free memory
    iftDestroyImage(&G1);
    iftDestroyImage(&G2);
    iftDestroyImage(&orig_altis);
    iftDestroyImage(&corr_altis);
    iftDestroyImage(&anomalies);

    // Saving output
    FILE *f = fopen(argv[4],"a+");
    if (f){
        char *s = iftBasename(argv[3]);
        fprintf(f,"%s;%.4f;%.4f;%.4f\n",s,p,perc_G1,perc_G2);
        free(s);
    }else{
        iftError("Failed to open file","main");
    }
    fclose(f);

    return 0;
}