
#include "cursor.h"


wxCursor *Image2Cursor(Image *img, Image *mask,
                       int hotSpotX, int hotSpotY,
                       int zoom, int color1, int color0)
{
    Image *zimg, *zmask;
    int q, p, nr, ncols, nrows, ncolsr, i, j;

    hotSpotX *= zoom;
    hotSpotY *= zoom;
    zimg  = Zoom(img, zoom, zoom);
    zmask = Zoom(mask, zoom, zoom);

    ncols = zimg->ncols;
    nrows = zimg->nrows;
    if (ncols != zmask->ncols || nrows != zmask->nrows)
        return NULL;

    ncolsr =  (ncols / 8 + 1) * 8;
    nr = ncolsr * nrows;

    char *down_bits = NULL, *down_mask = NULL;


    down_bits = AllocCharArray(nr / 8);
    down_mask = AllocCharArray(nr / 8);
#ifdef _WIN32
    for (i = 0; i < nr / 8; i++) down_bits[i] = 0x00; // 0=white 1=black
    for (i = 0; i < nr / 8; i++) down_mask[i] = 0xFF; // 1=transparent 0=solid
#endif
    q = 0;
    unsigned char selbit = 1;
    for (i = 0; i < nrows; i++)
    {
        for (j = 0; j < ncolsr; j++)
        {
            p = j + i * ncols;
            if (!ValidPixel(zimg, j, i))
            {
#ifndef _WIN32
                down_mask[q] &= ~selbit;
#endif
            }
            else
            {
                if (zimg->val[p] > 0)  down_bits[q] |= selbit;
                else                   down_bits[q] &= ~selbit;
                if (zmask->val[p] > 0) down_mask[q] |= selbit;
                else                down_mask[q] &= ~selbit;
            }
            if (selbit == 128)
            {
                q++;
                selbit = 1;
            }
            else selbit <<= 1;
        }
    }

    wxColour wxcolor1;
    wxColour wxcolor0;
    SetColor(&wxcolor1, color1);
    SetColor(&wxcolor0, color0);

#ifdef __WXMSW__
    wxBitmap down_bitmap(down_bits, ncolsr, nrows);
    wxBitmap down_mask_bitmap(down_mask, ncolsr, nrows);
    down_bitmap.SetMask(new wxMask(down_mask_bitmap));
    wxImage down_image = down_bitmap.ConvertToImage();
    down_image.SetOption(wxIMAGE_OPTION_CUR_HOTSPOT_X, hotSpotX);
    down_image.SetOption(wxIMAGE_OPTION_CUR_HOTSPOT_Y, hotSpotY);
    wxCursor *down_cursor = new wxCursor(down_image);

#elif __WXMAC__
    wxBitmap down_bitmap(down_bits, ncolsr, nrows);
    wxBitmap down_mask_bitmap(down_mask, ncolsr, nrows);
    down_bitmap.SetMask(new wxMask(down_mask_bitmap));
    wxImage down_image = down_bitmap.ConvertToImage();
    down_image.SetOption(wxIMAGE_OPTION_CUR_HOTSPOT_X, hotSpotX);
    down_image.SetOption(wxIMAGE_OPTION_CUR_HOTSPOT_Y, hotSpotY);
    wxCursor *down_cursor = new wxCursor(down_image);

#else
    wxCursor *down_cursor = new wxCursor(down_bits, ncolsr, nrows,
                                         hotSpotX, hotSpotY, down_mask,
                                         &wxcolor1, &wxcolor0);
#endif

    free(down_bits);
    free(down_mask);
    DestroyImage(&zimg);
    DestroyImage(&zmask);

    return down_cursor;
}



wxCursor *CrossCursor(int zoom)
{
    wxCursor *cursor = NULL;
    Image *img, *bin;
    int vimg[] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
                 };

    int vbin[] = {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0
                 };

    img = CreateImage(11, 11);
    bin = CreateImage(11, 11);

#ifdef _WIN32
    int i;
    for (i = 0; i < 11 * 11; i++)
    {
        if (vbin[i] == 0) vbin[i] = 1;
        else vbin[i] = 0;
        if (vimg[i] == 0) vimg[i] = 0;
        else vimg[i] = 255;
    }
#endif

    memcpy(img->val, vimg, sizeof(int) * 121);
    memcpy(bin->val, vbin, sizeof(int) * 121);


    cursor = Image2Cursor(img, bin, 5, 5, zoom,
                          0x000000, 0xFFFFFF);
    DestroyImage(&img);
    DestroyImage(&bin);

    return cursor;
}





