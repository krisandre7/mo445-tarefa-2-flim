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
	float avgx=0, avgy=0, avgz=0, avgn=0;

	for (int img=0; img < number_of_images; img++)
	{
		strcpy(full_path, directory);
		strcat(full_path, IN[img].image_name);
		iftImage * img = iftReadImage(full_path);
		avgx += img->xsize;
		avgy += img->ysize;
		avgz += img->zsize;
		avgn += img->n;
		iftDestroyImage(&img);
	}
	printf("Average x: %f\n", avgx/number_of_images*1.0);
	printf("Average y: %f\n", avgy/number_of_images*1.0);
	printf("Average z: %f\n", avgz/number_of_images*1.0);
	printf("Average voxels: %f\n", avgn/number_of_images*1.0);

	return 0;
}