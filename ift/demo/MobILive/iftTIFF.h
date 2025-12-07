#define TIFFGetR(abgr) ((abgr) & 0xff)
#define TIFFGetG(abgr) (((abgr) >> 8) & 0xff)
#define TIFFGetB(abgr) (((abgr) >> 16) & 0xff)
#define TIFFGetA(abgr) (((abgr) >> 24) & 0xff)

iftImage* TIFFRGB2IFT(char* filename)
{
  TIFF* tif = TIFFOpen(filename, "r");
  if (tif) {
    uint32 w, h;
    size_t npixels;
    uint32* raster;

    iftImage* img = NULL;
    
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    npixels = w * h;
    raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
    if (raster != NULL) {
      if (TIFFReadRGBAImage(tif, w, h, raster, 0)) {
	iftColor RGB,YCbCr;
	img = iftCreateColorImage(w,h,1);
	for(int s=0;s<img->n;s++) {
	  RGB.val[0] = TIFFGetR(raster[s]);
	  RGB.val[1] = TIFFGetG(raster[s]);
	  RGB.val[2] = TIFFGetB(raster[s]);
	  YCbCr      = iftRGBtoYCbCr(RGB);
	  //	  fprintf(stderr,"%d/%d",s,npixels-(1+(s/w))*w+(s%w));
	  img->val[npixels-(1+(s/w))*w+(s%w)] =        YCbCr.val[0];
	  //img->val[npixels-(1+(s/w))*w+(s%w)] = (RGB.val[0]+RGB.val[1]+RGB.val[2])/3;
	  img->Cb [npixels-(1+(s/w))*w+(s%w)] =(ushort)YCbCr.val[1];
	  img->Cr [npixels-(1+(s/w))*w+(s%w)] =(ushort)YCbCr.val[2];
	}
      }
      _TIFFfree(raster);
    }
    TIFFClose(tif);

    //iftWriteImageP6(img,"test.ppm");

    return img;
  } else {
    iftError("error while opening TIFF file","TIFFRGB2IFT");
  }

  return 0;
}
