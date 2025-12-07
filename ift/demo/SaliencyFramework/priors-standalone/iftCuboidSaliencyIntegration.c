#include "ift.h"

#include "iftGraphSaliency.c"

int main(int argc, const char *argv[]) {
    int number_maps = atoi(argv[1]);

    if (argc != number_maps + 5) {
        iftError(
                "Usage: iftCuboidSaliencyIntegration <number_of_maps[N]> <map0 map1 .... mapN> <iterations> <lambda> <output_file>",
                "main");
    }

    char input_file[250];
    iftImage **maps_list = (iftImage**) iftAlloc((size_t) number_maps, sizeof(iftImage*));
    for(int m = 0; m < number_maps; m++){
        strcpy(input_file, argv[m+2]);
        maps_list[m] = iftReadImageByExt(input_file);
    }


    int iterations = atoi(argv[number_maps+2]);
    float lambda = atof(argv[number_maps+3]);
    char output_file[250];
    strcpy(output_file, argv[number_maps+4]);

    iftAdjRel *A = iftCircular(sqrtf(2.0));
    iftImage *combined_maps = iftCuboidSaliencyIntegration(maps_list, number_maps, A, iterations, lambda);

    iftWriteImageByExt(combined_maps, output_file);

    iftDestroyAdjRel(&A);
    iftDestroyImage(&combined_maps);
    for(int r = 0; r < number_maps; r++) {
        iftImage *aux2 = maps_list[r];
        iftDestroyImage(&aux2);
    }
    iftFree(maps_list);
}