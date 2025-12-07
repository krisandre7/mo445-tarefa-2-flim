#include "ift.h"

int main(int argc, char **argv) 
{
	if (argc != 3)
		iftError("<labels_directory with /> <scn|pgm>", "Unify Labels - main");

	char *labels_directory = argv[1];
	char *extension = argv[2];

	int i, p, class, max=0, unique_number=0, number_of_images, number_of_images_folder;
	char *buffer=NULL, *unique_name=NULL, *str_class=NULL;
	char path_temp[256], full_name[256];
	float object_sizes[200];
	int max_class = 0;
	iftImage *label_image = NULL;

	number_of_images_folder = iftCountImageNames(labels_directory, extension);
	iftImageNames * image_names = iftCreateAndLoadImageNames(number_of_images_folder, labels_directory, extension);
	for (i=0; i < number_of_images_folder; i++)
	{
		buffer = iftSplitStringOld(image_names[i].image_name, ".", 0);
		unique_name = iftSplitStringOld(buffer, "_", 1);
		if (atoi(unique_name) > max)
			max = atoi(unique_name);
	}
	for(i=0; i< 200; i++) object_sizes[i] = 0;
	number_of_images = max+1;
	
	//Initialize
	iftImage **unified_labels = (iftImage**) malloc (sizeof(iftImage*) * (number_of_images));
	for (i=0; i < number_of_images; i++)
		unified_labels[i] = NULL;
	
	//Loading
	for (i=0; i < number_of_images_folder; i++)
	{
		buffer = iftSplitStringOld(image_names[i].image_name, ".", 0);
		unique_name = iftSplitStringOld(buffer, "_", 1);
		unique_number = atoi(unique_name);
		str_class = iftSplitStringOld(buffer, "_", 0);
		class = atoi(str_class);
		
		if(class > max_class)
			max_class = class;
		strcpy(full_name, labels_directory);
		strcat(full_name, image_names[i].image_name);
		if (strcmp(extension, "pgm") == 0) label_image = iftReadImageP5(full_name);
		else if (strcmp(extension, "scn") == 0) label_image = iftReadImage(full_name);
		else iftError("Extension not accepted. Please check the code", "main");
			
		for (p = 0; p < label_image->n ; p++)
		{
			if (label_image->val[p] != 0)	//If it is the object
				object_sizes[class-1]+=1;
		}
	}
	for (i=0; i< 200; i++)
	{
		if (object_sizes[i] != 0)
			object_sizes[i] = object_sizes[i] / (number_of_images_folder/max_class);
	}
	printf("Total images: %d\n", (number_of_images_folder/max_class));

	//Unify all the labels
	for (i=0; i < number_of_images_folder; i++)
	{
		printf("%d: Processing: %s\n", i, image_names[i].image_name);
		buffer = iftSplitStringOld(image_names[i].image_name, ".", 0);
		unique_name = iftSplitStringOld(buffer, "_", 1);
		unique_number = atoi(unique_name);
		str_class = iftSplitStringOld(buffer, "_", 0);
		class = atoi(str_class);
			
		strcpy(full_name, labels_directory);
		strcat(full_name, image_names[i].image_name);
		if (strcmp(extension, "pgm") == 0) label_image = iftReadImageP5(full_name);
		else if (strcmp(extension, "scn") == 0) label_image = iftReadImage(full_name);
		else iftError("Extension not accepted. Please check the code", "main");

		if (unified_labels[unique_number] == NULL)
			unified_labels[unique_number] = iftCreateImage(label_image->xsize, label_image->ysize, label_image->zsize);
			
		for (p = 0; p < label_image->n ; p++)
		{
			if (label_image->val[p] != 0)	//If it is the object
			{
				if (unified_labels[unique_number]->val[p] == 0)	//If its not assigned previously
				{
					unified_labels[unique_number]->val[p] = class;	//Puts in the output imagem the number of the object
				}
				else
				{
					//If the object present has more pixels, replace with the label of the smaller object
					if (object_sizes[unified_labels[unique_number]->val[p]-1] > object_sizes[class-1])
					{
						unified_labels[unique_number]->val[p] = class;
					}
						
				}
			}
		}
		iftDestroyImage(&label_image);

	}
	
	strcpy(full_name, "../base/");
	char *database = iftSplitStringOld(labels_directory, "/", 2);
	strcat(full_name, database);
	strcat(full_name, "/labels/unified/");
	iftCreateDirectory(full_name);
	puts(full_name);

	//Save on disk
	puts("Saving images");
	for (i=0; i < number_of_images; i++)
	{
		if (unified_labels[i] != NULL)
		{
			strcpy(path_temp, full_name);
			if (strcmp(extension, "pgm") == 0)
			{
				sprintf(buffer, "000001_%07d.pgm", i);
				strcat(path_temp, buffer);
				iftWriteImageP5(unified_labels[i], path_temp);
			}
			else if (strcmp(extension, "scn") == 0)
			{
				sprintf(buffer, "000001_%07d.scn", i);
				strcat(path_temp, buffer);
				iftWriteImage(unified_labels[i], path_temp);
			}
			iftDestroyImage(&unified_labels[i]);
		}
	}


	//Clean up
	free(unified_labels);
	return 0;
}
