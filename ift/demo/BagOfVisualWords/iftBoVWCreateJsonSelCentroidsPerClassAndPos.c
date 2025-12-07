//
// Created by Cesar Castelo on Feb 15, 2019
//

#include <ift.h>

int main(int argc, char** argv)
{
    if(argc != 5) {
        iftError("\nUsage: iftBoVWCreateJsonSelCentroidsPerClassAndPos <...>\n"
                         "[1] input_dir: Input directory containing the selected samples per class and position (JSON files)\n"
                         "[2] n_classes: Number of classes\n"
                         "[3] n_positions: Number of positions\n"
                         "[4] output_file: Output file containing all the selected centroids (.json)\n",
                 "iftBoVWCreateJsonSelCentroidsPerClassAndPos.c");
    }

    /* read parameters */
    char *inputDirname = iftCopyString(argv[1]);
    int nClasses = atoi(argv[2]);
    int nPositions = atoi(argv[3]);
    char *outputFilename = iftCopyString(argv[4]);

    /* read the input directory */
    iftFileSet *jsonFileset = iftLoadFileSetFromDirOrCSV(inputDirname, 1, true);
    iftDict *outDict = iftCreateDict();
    int nFiles = jsonFileset->n;

    if((nClasses*nPositions) != nFiles)
        iftError("The number of positions x number of classes and the number of files found does not match", "iftBoVWCreateJsonSelCentroidsPerClassAndPos");

    iftInsertIntoDict("hierarchy_type", "per_image_class_and_position", outDict);
    iftInsertIntoDict("n_classes", nClasses, outDict);
    iftInsertIntoDict("n_positions", nPositions, outDict);

    printf("--> %d files found in the input directory ...\n", nFiles);

    int nCentroids = 0, countCla = 0, countPos = 0;
    for(int i = 0; i < nFiles; i++) {
        if(!iftCompareStrings(iftFileExt(jsonFileset->files[i]->path), ".json"))
            iftError("One of the files in the input directory is not a JSON file", "iftBoVWCreateJsonSelCentroidsPerPos.c");
        
        /* read the dictionaty from the json file */
        iftDict *dict = iftReadJson(jsonFileset->files[i]->path);
        char *dataset = iftGetStrValFromDict("dataset", dict);
        iftIntArray *selSamples = iftGetIntArrayFromDict("selected_samples", dict);
        nCentroids += selSamples->n;

        /* get the class and the coordinates from the dataset filename */
        char *filename = iftFilename(jsonFileset->files[i]->path, ".json");
        int cla = -1, x = -1, y = -1;
        for(int c = 0; c < strlen(filename)-4; c++) {
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

            /* x coordinate */
            if(x == -1 &&
                filename[c] == 'x' &&
                isdigit(filename[c+1]) &&
                isdigit(filename[c+2]) &&
                isdigit(filename[c+3]) &&
                isdigit(filename[c+4]))
            {
                char *x_str = (char*)calloc(5, sizeof(char));
                memcpy(x_str, filename+c+1, 4);
                x_str[4] = '\0'; 
                x = atoi(x_str);
            }

            /* y coordinate */
            if(y == -1 &&
                filename[c] == 'y' &&
                isdigit(filename[c+1]) &&
                isdigit(filename[c+2]) &&
                isdigit(filename[c+3]) &&
                isdigit(filename[c+4]))
            {
                char *y_str = (char*)malloc(5);
                memcpy(y_str, filename+c+1, 4);
                y_str[4] = '\0'; 
                y = atoi(y_str);
            }
        }
        if(cla == -1 || x == -1 || y == -1)
            iftError("The dataset filename does not contain the clas number or the [x,y] coordinates", "iftBoVWCreateJsonSelCentroidsPerClassAndPos.c");

        /* insert the values in the output dictionary */
        char dictName[128];
        sprintf(dictName, "class_%d_position_%d", countCla+1, countPos+1);
        iftDict *posDict = iftCreateDict();
        iftInsertIntoDict("class", cla, posDict);
        iftInsertIntoDict("x", x, posDict);
        iftInsertIntoDict("y", y, posDict);
        iftInsertIntoDict("dataset", dataset, posDict);
        iftInsertIntoDict("selected_samples", selSamples, posDict);
        iftInsertIntoDict(dictName, posDict, outDict);

        countPos++;

        if(countPos == nPositions) {
            countCla++;
            countPos = 0;
        }
    }

    iftWriteJson(outDict, outputFilename);

    printf("--> JSON file created containing %d centroids for the %d classes and %d positions\n", nCentroids, nClasses, nPositions);

    iftDestroyFileSet(&jsonFileset);
    iftDestroyDict(&outDict);

    return 0;
}
