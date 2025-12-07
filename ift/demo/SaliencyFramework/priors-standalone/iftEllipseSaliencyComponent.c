
#include "ift.h"


void display_progress_bar(int progress, int total)
{
    printf("\r[");
    int pos = 50 * progress / total;

    for (int i = 0; i < 50; i++)
    {
        if (i < pos)
            printf("=");
        else if (i == pos)
            printf(">");
        else
            printf(" ");
    }

    printf("] %d%%", (int)(progress * 100.0 / total));
    fflush(stdout);
}

int main(int argc, char *argv[])
{

    if (argc != 3)
    {
        iftError("usage: iftEllipseSaliencyComponent [saliency_path] [output_path]. \nFor example, iftSaliencyEllipseFitting ./salie ./salie_post",
                 "iftSaliencyEllipseFitting");
    }

    char *path = argv[1];
    char *output_path = argv[2];
    char *filename = iftAllocCharArray(512);
    iftFileSet *fset = iftLoadFileSetFromDir(path, 1);
    // iftMakeDir(output_path);
    for (int i = 0; i < fset->n; i++)
    {
      display_progress_bar(i, fset->n);
      char *file = fset->files[i]->path;
      
      iftImage *salie_map      = iftReadImageByExt(file);
      iftImage *salie_map_filt =
	iftFilterSaliencyByEllipseMatching(salie_map, 32, 1000, 9000, 0.7);
      
      char *name = iftFilename(fset->files[i]->path, ".png");
      sprintf(filename, "%s/%s.png", output_path, name);
      iftWriteImageByExt(salie_map_filt, filename);
      
      iftDestroyImage(&salie_map);
      iftDestroyImage(&salie_map_filt);
    }
    iftFree(filename);
    iftDestroyFileSet(&fset);
    
    return 0;
}
