//
// Created by peixinho on 28/10/15.
//

#include "ift.h"

int main() {

    iftDir* dir = iftLoadFilesFromDirBySuffix("/home/peixinho/TrainPlates/orig/", "pgm");

    for (int i = 0; i < dir->nfiles; ++i) {
        iftImage* img = iftReadImageP5(dir->files[i]->pathname);

        for(int j = 0; j < 20; ++j) {

        }

        iftDestroyImage(&img);
    }

   return 0;
}