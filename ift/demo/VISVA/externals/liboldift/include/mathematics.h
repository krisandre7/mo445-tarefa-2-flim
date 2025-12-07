#ifndef _MATHEMATICS_H_
#define _MATHEMATICS_H_

#include "image.h"

Image  *Diff(Image *img1,Image *img2); 
Image  *Sum(Image *img1, Image *img2);
Image  *SQRT(Image *img);
Image  *And(Image *img1, Image *img2);
Image  *Or(Image *img1, Image *img2);
Image  *XOr(Image *img1, Image *img2);
Image  *Add(Image *img, int value);
Image  *Sub(Image *img, int value);
Image  *Complement(Image *img);
Image  *Div(Image *img1,Image *img2);
Image  *Mult(Image *img1,Image *img2);
Image  *Abs(Image *img);
Image  *Average(Image *img1, Image *img2);
Image  *Log(Image *img);

void Diffinplace(Image *img1,Image *img2);
void Suminplace(Image *img1, Image *img2);
void Orinplace(Image *img1,Image *img2);

// Mathematical functions with no images

double Factorial(int n);
double DoubleFactorial(int n);
double HypersphereVolume(int dim, float radius);



#endif


