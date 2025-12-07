#include "ift.h"

int main(int argc, char **argv)
{
    if (argc != 3)
        iftError("<orig_directory with /> <scn|ppm|pgm>", "Database Depth - main");

    char *labels_directory = argv[1];
    char *extension = argv[2];

    int i, number_of_images, max_depth, curr_depth;
    char path_temp[512];
    iftImage *image = NULL;

    number_of_images = iftCountImageNames(labels_directory, extension);
    iftImageNames *image_names = iftCreateAndLoadImageNames(number_of_images, labels_directory, extension);
    max_depth = -1;

    for (i = 0; i < number_of_images; i++)
    {
        strcpy(path_temp, labels_directory);
        strcat(path_temp, image_names[i].image_name);
        if (strcmp(extension, "scn") == 0)
            image = iftReadImage(path_temp);
        else if (strcmp(extension, "ppm") == 0)
            image = iftReadImageP6(path_temp);
        else
            image = iftReadImageP5(path_temp);

        curr_depth = iftRadiometricResolution(image);
        if (curr_depth > max_depth)
            max_depth = curr_depth;

        iftDestroyImage(&image);
    }
    printf("Dataset depth: %d\n", max_depth);

    return 0;
}
