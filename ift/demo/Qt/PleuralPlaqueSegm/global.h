#ifndef GLOBAL_H
#define GLOBAL_H

#include <QMainWindow>
extern "C" {
#include "ift.h"
}

/**
 * @brief Returns a QImage generated from a iftImage
 *
 * @author Azael de Melo e Sousa
 *
 * Transforms a iftImage into a QImage
 *
 * @param <img> Pointer to an image stored in a iftImage data structure
 * @return Same image stored in a QImage data structure
 */
QImage iftImage2QImage(iftImage *img);

/**
 * @brief Returns a iftImage generated from a QImage
 *
 * @author Azael de Melo e Sousa
 *
 * Transforms a QImage into a iftImage
 *
 * @param <img> Image stored in a QImage data structure
 * @return Pointer to a copy of the QImage in a iftImage
 */
iftImage* iftQImage2Image(QImage img);

/**
 * @brief Calculate the contrast of a QImage
 *
 * @author Azael de Melo e Sousa
 *
 * Changing the contrast of a image, changes the range of luminance
 * values present. In the histogram it is equivalent to expanding or
 * compressing the histogram around the midpoint value. Mathematically
 * it is expressed as:
 *
 * new_value = (old_value - 0.5) × contrast + 0.5
 *
 * @param <&img>    Reference to an image stored in a QImage data structure
 *        <factor>  Constant used to calculate the contrast
 * @return void
 */
void qContrast(QImage &img, double factor);

/**
 * @brief Calculate the brightness of a QImage
 *
 * @author Azael de Melo e Sousa
 *
 * When changing the brightness of an image, a constant is added or
 * subtracted from the luminnance of all sample values. This is equivalent
 * to shifting the contents of the histogram left (subtraction) or right
 * (addition).
 *
 * new_value = old_value + brightness
 *
 * @param <&img>   Reference to an image stored in a QImage data structure
 *        <shift>  Constant used to calculate the brightness
 * @return void
 */
void qBrightness(QImage &img, int shift);

/**
 * @brief Normalize a QImage
 *
 * @author Azael de Melo e Sousa
 *
 * Normalize an image according to a normalization value
 *
 * @param <&img>   Reference to an image stored in a QImage data structure
 *        <norm>   normalization value
 * @return void
 */
void qNormalizeImage(QImage &img, int norm);

/**
 * @brief Find the minimum value in a QImage
 *
 * @author Azael de Melo e Sousa
 *
 * Scan all image to find its minimum value
 *
 * @param <&img>   Reference to an image stored in a QImage data structure
 * @return Minimum value (uint)
 */
uint qMinimumValue(QImage &img);

/**
 * @brief Find the minimum value in a QImage
 *
 * @author Azael de Melo e Sousa
 *
 * Scan all image to find its minimum value
 *
 * @param <&img>   Reference to an image stored in a QImage data structure
 * @return Minimum value (uint)
 */
uint qMaximumValue(QImage &img);

#endif // GLOBAL_H
