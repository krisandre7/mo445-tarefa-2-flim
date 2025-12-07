#include <iftMemory.h>
#include "ift.h"

void PrintClassHistogram(iftDataSet *Z)
{
    int c, s, *nclass=iftAllocIntArray(Z->nclasses);

    for (s=0; s < Z->nsamples; s++)
        nclass[Z->sample[s].truelabel-1]++;

    for (c=0; c < Z->nclasses; c++)
        printf("class %d: %d\n",c+1,nclass[c]);

    free(nclass);

}

iftDict *iftGetArguments(int argc, const char **argv);

int main(int argc, const char **argv)
{
    iftDataSet      *Z=NULL;
    iftCplGraph     *graph=NULL;
    timer           *t1=NULL,*t2=NULL;
    float            acc = 0.0;
    iftDataSet      *Z_test = NULL;
    iftDict         *args = NULL;
    iftJson         *config = NULL;
    double           perc_train = 0.2;

    /*--------------------------------------------------------*/

    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/

    iftRandomSeed(IFT_RANDOM_SEED);

    args = iftGetArguments(argc, argv);

    char *input_dataset  = iftGetStrValFromDict("--input-training-dataset", args);
    char *output_classif = iftGetStrValFromDict("--ouptut-classifier", args);
    char *config_file    = iftGetStrValFromDict("--input-config", args);

    if(!iftFileExists(input_dataset))
        iftError("Dataset file %s does not exist!", "main", input_dataset);

    if(!iftFileExists(config_file))
        iftError("Json configuration file %s does not exist!", "main", config_file);


    /* Initialization */

    Z  = iftReadOPFDataSet(input_dataset); // Read dataset Z

    config = iftReadJson(config_file);

    perc_train = iftGetJDouble(config, "perc_training");

    printf("Distance function number %d\n", Z->function_number);

    printf("Total number of samples  %d\n",Z->nsamples);
    printf("Total number of features %d\n",Z->nfeats);
    printf("Total number of classes %d\n",Z->nclasses);

    t1     = iftTic();

    /* Conceito basico de classificacao de padroes. F eh um classificador de padroes, que nada mais eh do que uma funcao
     * que recebe um vetor de caracteristicas x representando uma amostra, e retorna um valor binario indicando se ele
     * pertence a classe 1 ou a classe 2. Ou seja:
     *
     * F(Z, x) -> 2, 1 x -> Z eh o dataset de treinamento, x eh o vetor de caracteristicas da tile, F eh o classificador,
     */
    iftSelectSupTrainSamples(Z, perc_train); // Select training samples

    // Perform supervised learning
    graph = iftSupLearn(Z);
    Z_test = iftExtractSamples(Z, IFT_TEST);
    iftClassify(graph, Z_test);                      // Classify test samples
    acc = iftTruePositives(Z_test);        // Compute accuracy on test set

    t2     = iftToc();

    fprintf(stdout,"Classifier training\n");
    iftPrintFormattedTime(iftCompTime(t1,t2));

    fprintf(stdout,"Accuracy of classification is %f\n",acc);

    iftWriteCplGraph(graph, output_classif);

    iftDestroyCplGraph(&graph);

    /* Cleaning up! */
    iftDestroyDataSet(&Z_test);
    iftDestroyDataSet(&Z);

    iftDestroyJson(&config);
    iftDestroyMinkowskiTable();
    iftDestroyDict(&args);
    free(input_dataset);
    free(output_classif);
    free(config_file);

    /* ---------------------------------------------------------- */


    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

    return(0);
}





iftDict *iftGetArguments(int argc, const char **argv) {
    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-training-dataset", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Input dataset for training"},
            {.short_name = "-c", .long_name = "--input-config", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Json config file"},
            {.short_name = "-o", .long_name = "--ouptut-classifier", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="The output dataset filename"},
    };

    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    char program_description[2048] = "This is program takes as input a pre-computed dataset and trains an SVM classifier on it.";

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);

    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

    iftDestroyCmdLineParser(&parser);

    return args;
}
