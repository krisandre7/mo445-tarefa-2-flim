#include "ift.h"

int main(int argc, char **argv) 
{
	if (argc != 2)
		iftError("<scn direcotry>", "SCN to txt - main");

	char * directory = argv[1];
	int number_of_images = iftCountImageNames(directory, "scn");
	iftImageNames *IN = iftCreateAndLoadImageNames(number_of_images, directory, "scn");
	char full_path[256];
	int i;

	for (int img=0; img < number_of_images; img++)
	{
		strcpy(full_path, "txts/");
		strcat(full_path, IN[img].image_name);
		strcat(full_path, ".txt");
		FILE *output = fopen(full_path, "w");
		if (!output) iftError("File not open", "SCN to txt - main");

		strcpy(full_path, directory);
		strcat(full_path, IN[img].image_name);
		printf("Processing: %s\n", full_path);
		iftImage * img = iftReadImage(full_path);
		for (i=0; i< img->n; i++)
			fprintf(output, "%d ", img->val[i]);
		iftDestroyImage(&img);

		fclose(output);
	}
	return 0;
}