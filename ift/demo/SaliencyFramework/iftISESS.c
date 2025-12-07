#include "ift.h"
#include "iftSaliency.h"

int main(int argc, const char *argv[]) {
    /* General variables */
    char orig_file[250], initial_saliency_file[250], output_file[250];
    iftImage *orig, *initial_sal, *final_sal;

    if (argc != 4) {
        iftError("Usage: iftSESS <orig_image> <initial_saliency> <output_saliency>", "main");
    }

    strcpy(orig_file, argv[1]);
    strcpy(initial_saliency_file, argv[2]);
    strcpy(output_file, argv[3]);

    orig = iftReadImageByExt(orig_file);
    initial_sal = iftReadImageByExt(initial_saliency_file);
    iftITSELFParameters *params = iftInitializeITSELFParametersByDefaultU2Net();

    final_sal = iftSESS(orig, initial_sal, params, NULL);

    iftDestroyITSELFParameters(&params);

    iftWriteImageByExt(final_sal, output_file);


    iftDestroyImage(&orig);
    iftDestroyImage(&initial_sal);
    iftDestroyImage(&final_sal);

    return 0;
}