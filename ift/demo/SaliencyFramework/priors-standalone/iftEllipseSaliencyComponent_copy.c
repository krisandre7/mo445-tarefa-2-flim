
#include "ift.h"

static iftImage *iftEllipseMatchingObject(iftImage *superpixel_img, iftTensorScale *tensor_scale, iftFloatArray *ellipseMatch, float variance, int min_size, int max_size)
{
    iftIntArray *region_sizes = iftCountLabelSpels(superpixel_img);

    float min_ellipse_score = IFT_INFINITY_FLT;
    for (int s = 0; s < ellipseMatch->n; s++)
    {
        if (tensor_scale->area->val[s] >= (float)min_size || tensor_scale->area->val[s] <= (float)max_size)
        {
            ellipseMatch->val[s] /= (float)region_sizes->val[s + 1];
            //            printf("Match = %f\n", ellipseMatch->val[s]);
            ellipseMatch->val[s] = expf(ellipseMatch->val[s] / variance);
            if (ellipseMatch->val[s] < min_ellipse_score)
                min_ellipse_score = ellipseMatch->val[s];
        }
    }

    for (int s = 0; s < ellipseMatch->n; s++)
    {
        if (tensor_scale->area->val[s] < (float)min_size || tensor_scale->area->val[s] > (float)max_size)
            ellipseMatch->val[s] = min_ellipse_score;
    }

    float max = 0;
    float min = IFT_INFINITY_FLT;
    for (int rj = 0; rj < ellipseMatch->n; rj++)
    { // Get Max
        if (ellipseMatch->val[rj] > max)
            max = ellipseMatch->val[rj];
        if (ellipseMatch->val[rj] < min)
            min = ellipseMatch->val[rj];
    }

    for (int rj = 0; rj < ellipseMatch->n; rj++)
        ellipseMatch->val[rj] = (ellipseMatch->val[rj] - min) / (max - min);

    iftDestroyIntArray(&region_sizes);

    iftImage *ellipseMatchImage = iftCreateImageFromImage(superpixel_img);
    for (int k = 0; k < superpixel_img->n; k++)
        ellipseMatchImage->val[k] = (int)(255 * ellipseMatch->val[superpixel_img->val[k] - 1]);

    return ellipseMatchImage;
}

/**
 * @brief Find the ellipse of an saliency grey scale image using Tensor Scale.
 *
 * This function estimate an ellipse for each region of object and
 * region of background in the mask.
 * @param salie_map The saliency mask.
 * @param tensor_ratios The pairs of radially opposite sample lines, that are traced emerging from
                        the center pixel.
 * @param min_size minimum size of region label computation.
 * @param max_size maximum size of region label computation.
 * @param score_threshold the threshold (or overllaping confidence score) of FScore estimated after ellipse matching
 * @return void
 */
void iftFilterSaliencyByEllipseMatchingInPlace(iftImage *salie_map, int tensor_ratios, int min_size, int max_size, float score_threshold)
{
    iftImage *salie_map_bin = iftBinarize(salie_map);
    // this filter mask is a boolean image used to zero out the saliency map values
    iftImage *filter_mask = iftCreateImage(salie_map->xsize, salie_map->ysize, salie_map->zsize);
    // label its components from 1 to n
    iftImage *salie_map_bin_label = iftFastLabelComp(salie_map_bin, NULL);
    int nobjs = iftMaximumValue(salie_map_bin_label);
    int o, p, q, r;
    char *filename = iftAllocCharArray(512);
    // for each component
    for (o = 1; o <= nobjs; o++)
    {
        // extract the component from the label image
        iftImage *obj = iftExtractObject(salie_map_bin_label, o);

        // estimate binary area
        int object_area = 0;
        for (int a = 0; a < obj->n; a++)
        {
            if (obj->val[a] > 0)
            {
                object_area = object_area + 1;
            }
        }

        if (object_area > 100) // avoid small objects 
        {
            // relabel it as though it were a superpixel image
            for (p = 0; p < obj->n; p++)
            {
                if (obj->val[p] != 0)
                    obj->val[p] = 2;
                else
                    obj->val[p] = 1;
            }

            iftTensorScale *tensor_scale = iftSuperpixelToTensorScale(obj, tensor_ratios, min_size, max_size); // ISSUE: MEMORY LEAK
            // printf("Tensor Area range: [%f, %f]\n", tensor_scale->area->val[1], tensor_scale->area->val[0]);

            iftVoxelArray *centroids = iftGeometricCentersFromLabelImage(obj);
            iftFloatArray *ellipseMatch = iftCreateFloatArray(centroids->n - 1);
            for (int p = 0; p < obj->n; p++)
            {
                iftVoxel u = iftGetVoxelCoord(obj, p);
                int s = obj->val[p] - 1;
                float ellipse_eq = sqrtf(square((u.x - tensor_scale->pos_focus->val[s].x)) + (square((u.y - tensor_scale->pos_focus->val[s].y)))) + sqrtf(square((u.x - tensor_scale->neg_focus->val[s].x)) + (square((u.y - tensor_scale->neg_focus->val[s].y))));
                if (ellipse_eq <= 2 * tensor_scale->major_axis->val[s])
                {
                    ellipseMatch->val[s] += 1;
                }
            }
            iftImage *ellipseMatchImage = iftEllipseMatchingObject(obj, tensor_scale, ellipseMatch, 0, min_size, max_size);
            iftDestroyTensorScale(&tensor_scale);
            //  compute overllaping score
            iftImage *a = iftBinarize(ellipseMatchImage);
            iftImage *b = iftBinarize(obj);
            float score = iftFScore(a, b);
            // printf("[INFO] Score: %f\n", score);
            iftDestroyImage(&a);
            iftDestroyImage(&b);
            iftDestroyImage(&ellipseMatchImage);
            iftDestroyFloatArray(&ellipseMatch);

            if (score > score_threshold)
            {

                iftImage *norm = iftNormalize(obj, 0, 255);

                for (q = 0; q < norm->n; q++)
                {
                    if (norm->val[q] > 0)
                    {
                        filter_mask->val[q] = 1;
                    }
                }

                iftDestroyImage(&norm);
            }
            iftDestroyImage(&object);
        }
        else
        {
            iftDestroyImage(&obj);
            continue;
        }
    }

    for (r = 0; r < filter_mask->n; r++)
    {
        if (filter_mask->val[r] < 1)
        {
            salie_map->val[r] = 0;
        }
    }

    iftDestroyImage(&filter_mask);
    iftDestroyImage(&salie_map_bin);
    iftDestroyImage(&salie_map_bin_label);
    iftFree(filename);
}

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
     
        iftImage *salie_map = iftReadImageByExt(file);
        iftFilterSaliencyByEllipseMatchingInPlace(salie_map, 32, 0, 1000, 0.5);

        char *name = iftFilename(fset->files[i]->path, ".png");
        sprintf(filename, "%s/%s.png", output_path, name);
        iftWriteImageByExt(salie_map, filename);

        iftDestroyImage(&salie_map);
    }
    iftFree(filename);
    iftDestroyFileSet(&fset);
    
    return 0;
}
