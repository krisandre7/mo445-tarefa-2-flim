//
// Created by azaelmsousa on 11/11/20.
//

#include "ift.h"

int main(int argc, char *argv[]){

    if (argc != 3){
        iftError("Usage: iftImageDirIntoMImage <...>\n" \
                 "[1] Dir with images to be turned into a single mimage.\n" \
                 "[2] Output mimage.\n","main");
    }

    iftFileSet *fs = iftLoadFileSetFromDirOrCSV(argv[1],1,TRUE);
    iftMImage *out = NULL;

    for(int i = 0; i < fs->n; i++){
        iftImage *band = iftReadImageByExt(fs->files[i]->path);
        iftExtendMImageByImageInPlace(out,band);
        iftDestroyImage(&band);
    }
    iftDestroyFileSet(&fs);

    iftWriteMImage(out,argv[2]);
    iftDestroyMImage(&out);

    return 0;
}