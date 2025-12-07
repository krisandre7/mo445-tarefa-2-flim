//
// Created by azaelmsousa on 27/04/21.
//

#include "ift.h"

int main(int argc, char *argv[]){

    if (argc < 3){
        iftError("Usage: iftRegistrationErrorsComparison <...>\n[1..N] Input dirs with resulting reference image and respective error (.json).\n" \
                 "These files are the output of the program iftRegisterImageIntoNormalSpace.c\n" \
                 "The system requires at least 2 dirs.","main");
    }

    iftFileSet **fs_json = NULL;
    iftFileSet **fs_npy = NULL;
    fs_json = (iftFileSet**)calloc(sizeof(iftFileSet*),argc-1);
    fs_npy = (iftFileSet**)calloc(sizeof(iftFileSet*),argc-1);
    iftFloatArray *mean_error_per_dir = iftCreateFloatArray(argc-1);
    int *index = (int*)calloc(sizeof(int),argc-1);

    for (int i = 0; i < argc-1; i++){
        printf("Loading files from dir \"%s\"\n",argv[i+1]);
        index[i] = i+1;
        fs_json[i] = iftLoadFileSetFromDirBySuffix(argv[i+1],".json",1);
        fs_npy[i] = iftLoadFileSetFromDirBySuffix(argv[i+1],".npy",1);
        for (int f = 0; f < fs_json[i]->n; f++) {
            puts(iftBasename(fs_json[i]->files[f]->path));
            iftDict *json = iftReadJson(fs_json[i]->files[f]->path);
            iftFloatArray *errors = iftReadFloatArray(fs_npy[i]->files[f]->path);
            int error_id = iftGetLongValFromDict("registration-error-id",json);
            mean_error_per_dir->val[i] += errors->val[error_id];
            iftDestroyFloatArray(&errors);
            iftDestroyDict(&json);
        }
        mean_error_per_dir-val[i] /= fs_json[i]->n;
        iftDestroyFileSet(&fs_json[i]);
        iftDestroyFileSet(&fs_npy[i]);
    }
    iftFree(fs_json);
    iftFree(fs_npy);
    puts("");

    iftFQuickSort(mean_error_per_dir->val,index,0,(argc-1)-1,IFT_INCREASING);

    iftPrintFloatArray(mean_error_per_dir->val,mean_error_per_dir->n);
    iftDestroyFloatArray(&mean_error_per_dir);

    for(int i = 0; i < argc-1; i++)
        printf("%s\n",argv[index[i]]);
    iftFree(index);

    return 0;

}