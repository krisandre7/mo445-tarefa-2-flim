//
// Created by Cesar Castelo on Feb 25, 2019
//

#include <ift.h>

int main(int argc, char** argv)
{
    if(argc != 4) {
        iftError("\nUsage: iftBoVWCreateJsonSelCentroidsPerClass <...>\n"
                         "[1] input_dir: Input directory containing the selected samples per position (JSON files)\n"
                         "[2] n_classes: Number of classes\n"
                         "[3] output_file: Output file containing all the selected centroids (.json)\n",
                 "iftBoVWCreateJsonSelCentroidsPerClass.c");
    }

    /* read parameters */
    char *inputDirname = iftCopyString(argv[1]);
    int nClasses = atoi(argv[2]);
    char *outputFilename = iftCopyString(argv[3]);

    /* read the input directory */
    iftFileSet *jsonFileset = iftLoadFileSetFromDirOrCSV(inputDirname, 1, true);
    iftDict *outDict = iftCreateDict();
    int nFiles = jsonFileset->n;

    if(nClasses != nFiles)
        iftError("The number of classes and the number of files found does not match", "iftBoVWCreateJsonSelCentroidsPerClass");

    iftInsertIntoDict("hierarchy_type", "per_image_class", outDict);
    iftInsertIntoDict("n_classes", nClasses, outDict);

    printf("--> %d files found in the input directory ...\n", nFiles);

    int nCentroids = 0;
    for(int i = 0; i < nFiles; i++) {
        if(!iftCompareStrings(iftFileExt(jsonFileset->files[i]->path), ".json"))
            iftError("One of the files in the input directory is not a JSON file", "iftBoVWCreateJsonSelCentroidsPerClass.c");
        
        /* read the dictionaty from the json file */
        iftDict *dict = iftReadJson(jsonFileset->files[i]->path);
        char *dataset = iftGetStrValFromDict("dataset", dict);
        iftIntArray *selSamples = iftGetIntArrayFromDict("selected_samples", dict);
        nCentroids += selSamples->n;

        /* get the class from the dataset filename */
        char *filename = iftFilename(jsonFileset->files[i]->path, ".json");
        int cla = -1;
        for(int c = 0; c < strlen(filename)-7; c++) {
            /* class */
            if(cla == -1 &&
                !strncmp(filename+c, "class", 5) &&
                isdigit(filename[c+5]) &&
                isdigit(filename[c+6]) &&
                isdigit(filename[c+7]))
            {
                char *class_str = (char*)calloc(4, sizeof(char));
                memcpy(class_str, filename+c+5, 3);
                class_str[3] = '\0'; 
                cla = atoi(class_str);
            }
        }
        if(cla == -1)
            iftError("The dataset filename does not contain the class number", "iftBoVWCreateJsonSelCentroidsPerClass.c");

        /* insert the values in the output dictionary */
        char dictName[128];
        sprintf(dictName, "class_%d", i+1);
        iftDict *posDict = iftCreateDict();
        iftInsertIntoDict("dataset", dataset, posDict);
        iftInsertIntoDict("selected_samples", selSamples, posDict);
        iftInsertIntoDict(dictName, posDict, outDict);
    }

    iftWriteJson(outDict, outputFilename);

    printf("--> JSON file created containing %d centroids for the %d classes\n", nCentroids, nClasses);

    iftDestroyFileSet(&jsonFileset);
    iftDestroyDict(&outDict);

    return 0;
}
