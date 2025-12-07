//
// Created by Cesar Castelo on Feb 15, 2019
//

#include <ift.h>

int main(int argc, char** argv)
{
    if(argc != 4) {
        iftError("\nUsage: iftBoVWCreateJsonSelCentroidsPerPos <...>\n"
                         "[1] input_dir: Input directory containing the selected samples per position (JSON files)\n"
                         "[2] n_positions: Number of positions\n"
                         "[3] output_file: Output file containing all the selected centroids (.json)\n",
                 "iftBoVWCreateJsonSelCentroidsPerPos.c");
    }

    /* read parameters */
    char *inputDirname = iftCopyString(argv[1]);
    int nPositions = atoi(argv[2]);
    char *outputFilename = iftCopyString(argv[3]);

    /* read the input directory */
    iftFileSet *jsonFileset = iftLoadFileSetFromDirOrCSV(inputDirname, 1, true);
    iftDict *outDict = iftCreateDict();
    int nFiles = jsonFileset->n;

    if(nPositions != nFiles)
        iftError("The number of positions and the number of files found does not match", "iftBoVWCreateJsonSelCentroidsPerPos");

    iftInsertIntoDict("hierarchy_type", "per_position", outDict);
    iftInsertIntoDict("n_positions", nPositions, outDict);

    printf("--> %d files found in the input directory ...\n", nFiles);

    int nCentroids = 0;
    for(int i = 0; i < nFiles; i++) {
        if(!iftCompareStrings(iftFileExt(jsonFileset->files[i]->path), ".json"))
            iftError("One of the files in the input directory is not a JSON file", "iftBoVWCreateJsonSelCentroidsPerPos.c");
        
        /* read the dictionaty from the json file */
        iftDict *dict = iftReadJson(jsonFileset->files[i]->path);
        char *dataset = iftGetStrValFromDict("dataset", dict);
        iftIntArray *selSamples = iftGetIntArrayFromDict("selected_samples", dict);
        nCentroids += selSamples->n;

        /* get the coordinates from the dataset filename */
        char *filename = iftFilename(jsonFileset->files[i]->path, ".json");
        int x = -1, y = -1;
        for(int c = 0; c < strlen(filename)-4; c++) {
            /* x coordinate */
            if(filename[c] == 'x' &&
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
            if(filename[c] == 'y' &&
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
        if(x == -1 || y == -1)
            iftError("The dataset filename does not contain the [x,y] coordinates", "iftBoVWCreateJsonSelCentroidsPerPos.c");

        /* insert the values in the output dictionary */
        char dictName[128];
        sprintf(dictName, "position_%d", i+1);
        iftDict *posDict = iftCreateDict();
        iftInsertIntoDict("x", x, posDict);
        iftInsertIntoDict("y", y, posDict);
        iftInsertIntoDict("dataset", dataset, posDict);
        iftInsertIntoDict("selected_samples", selSamples, posDict);
        iftInsertIntoDict(dictName, posDict, outDict);
    }

    iftWriteJson(outDict, outputFilename);

    printf("--> JSON file created containing %d centroids for the %d positions\n", nCentroids, nFiles);

    iftDestroyFileSet(&jsonFileset);
    iftDestroyDict(&outDict);

    return 0;
}
