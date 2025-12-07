#include "ift.h"

int main(int argc, char **argv)
{
    if (argc != 2)
        iftError("<labels_directory with />", "Separate Objects - main");

    char *labels_directory = argv[1];
    char *extension = "scn";

    int i, o, p, number_of_objects = 0, number_of_images, *object_histogram;
    char buffer[512], *image_class = NULL, *image_id = NULL;
    char path_temp[256], full_name[256];
    iftImage *label_image = NULL, *tmp_image=NULL;

    number_of_images = iftCountImageNames(labels_directory, extension);
    iftImageNames *image_names = iftCreateAndLoadImageNames(number_of_images, labels_directory, extension);


    for (i = 0; i < number_of_images; i++)
    {
        strcpy(path_temp, labels_directory);
        strcat(path_temp, image_names[i].image_name);
        label_image = iftReadImage(path_temp);

        image_class = iftSplitStringOld(image_names[i].image_name, "_", 0);
        image_id = iftSplitStringOld(image_names[i].image_name, "_", 1);
        image_id = iftSplitStringOld(image_id, ".", 0);

        number_of_objects = iftMaximumValue(label_image);
        object_histogram = iftAllocIntArray(number_of_objects + 1);
        memset(object_histogram, 0, sizeof(int)*number_of_objects+1);
        for (o = 1; o <= number_of_objects; o++)
        {
            for (p = 0; p < label_image->n; p++)
            {
            	if (label_image->val[p] == o)
            		object_histogram[o]++;
            }
            if (object_histogram[o])
            {
            	tmp_image = iftCreateImage(label_image->xsize, label_image->ysize, label_image->zsize);
            	for (p = 0; p < label_image->n; p++)
            	{
	            	if (label_image->val[p] == o)
    	        		tmp_image->val[p] = label_image->val[p];
            	}

            	strcpy(path_temp, labels_directory);
            	strcat(path_temp, "separated/");
            	iftCreateDirectory(path_temp);

            	sprintf(buffer, "%05d_%s.scn", o, image_id);
            	strcpy(path_temp, labels_directory);
            	strcat(path_temp, "separated/");
            	strcat(path_temp, buffer);
            	puts(path_temp);
            	iftWriteImage(tmp_image, path_temp);
            	iftDestroyImage(&tmp_image);
            }
       }
       free(object_histogram);

    }

    return 0;
}
