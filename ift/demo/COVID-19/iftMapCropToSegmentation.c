//
// Created by azaelmsousa on 08/10/20.
//

#include "ift.h"

int main(int argc, char *argv[]){

    if (argc != 5)
        iftError("Usage: iftMapCropToSegmentation <segmentation of lungs> <right lung crop> <left lung crop> <output mask>","main");

    iftImage *segm = iftReadImageByExt(argv[1]);
    iftImage *rl = iftReadImageByExt(argv[2]);
    iftImage *ll = iftReadImageByExt(argv[3]);
    iftImage *out = iftCopyImage(segm);
    int last_label = iftMaximumValue(out);

    iftVoxel u,v,w;
    iftImage *segm_rl = iftExtractObject(segm,1);
    iftBoundingBox bb = iftMinBoundingBox(segm_rl,&u);
    v.x = 0;
    v.y = 0;
    v.z = 0;
    for (int z = bb.begin.z; z < bb.end.z; z++){
        v.y = 0;
        for (int y = bb.begin.y; y < bb.end.y; y++){
            v.x = 0;
            for (int x = bb.begin.x; x < bb.end.x; x++){
                w.x = x;
                w.y = y;
                w.z = z;
                if ((iftValidVoxel(rl,v)) && (iftValidVoxel(segm_rl,w))){
                    int p = iftGetVoxelIndex(segm_rl,w);
                    int q = iftGetVoxelIndex(rl,v);
                    if ((rl->val[q] > 0) && (out->val[p] > 0) && (out->val[p] < 3))
                        out->val[p] = last_label+rl->val[q];
                }
                v.x++;
            }
            v.y++;
        }
        v.z++;
    }
    iftDestroyImage(&segm_rl);
    iftDestroyImage(&rl);

    iftImage *segm_ll = iftExtractObject(segm,2);
    bb = iftMinBoundingBox(segm_ll,&u);
    v.x = 0;
    v.y = 0;
    v.z = 0;
    for (int z = bb.begin.z; z < bb.end.z; z++){
        v.y = 0;
        for (int y = bb.begin.y; y < bb.end.y; y++){
            v.x = 0;
            for (int x = bb.begin.x; x < bb.end.x; x++){
                w.x = x;
                w.y = y;
                w.z = z;
                if ((iftValidVoxel(ll,v)) && (iftValidVoxel(segm_ll,w))){
                    int p = iftGetVoxelIndex(segm_ll,w);
                    int q = iftGetVoxelIndex(ll,v);
                    if ((ll->val[q] > 0) && (out->val[p] > 0) && (out->val[p] < 3))
                        out->val[p] = last_label+ll->val[q];
                }
                v.x++;
            }
            v.y++;
        }
        v.z++;
    }
    iftDestroyImage(&segm_ll);
    iftDestroyImage(&ll);

    iftDestroyImage(&segm);

    iftWriteImageByExt(out,argv[4]);
    iftDestroyImage(&out);

    return 0;
}