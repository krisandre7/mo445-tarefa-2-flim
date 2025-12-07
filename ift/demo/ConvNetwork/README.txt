How to make:
	make convnet

How to run: (default parameters)

-----------------------------------------------------
---> Face database pre-processing: <---

Renaming and converting all the jpg files to pgm files:
url: http://www.ic.unicamp.br/~afalcao/mo443/projeto1s2013.tar.bz2

build the directory orig inside pubfig83 dataset and move all images inside it.
run the python script demo/ConvNetwork/scripts/base_renaming.py:

	python base_renaming.py <database_directory>
	
	ex: 001_100.jpg ---> 000001_00000111.pgm

-------------------------------------------------------

How to generate the file_names file with the expected format for the convnet app:
	python split_database.py <image_base_location> 83 100

The file will be saved inside the folder with the name: filenames.txt. This file will contain 83 classes with 100 samples per class.
This can also be used to split the database. Testing for example.

-----------------------------------------------------

SVM parameters suggestion:
 ./iftDataClassifyBySVM dataset.data 0.9 1 0 0 0 1

-----------------------------------------------------
How to convert all the jpeg files in a folder to pgm files:
	cd <FOLDERLOCATION>;
	for f in `find . -name "*.jpg"`; do     convert $f -resize 200x200 $f.pgm; done

-----------------------------------------------------

CONFIG FILE FORMAT:
For unidimensional parameters
<parameter>, <value>

For multidimensional parameters
<parameter>, <value>, <value>, <...>

N_LAYERS, 3
N_BANDS_INPUT, 1
INPUT_NORM_SIZE, 3
N_KERNELS, 64, 128, 256
SIZE_KERNELS, 3, 5, 7
SIZE_POOLING, 3, 5, 7
STRIDE, 2, 2, 2
ALPHA, 2, 2, 2
SIZE_NORM, 5, 7, 9

Good architecture
N_LAYERS, 3
N_BANDS_INPUT, 1
INPUT_NORM_SIZE, 9
N_KERNELS, 64, 128, 256
SIZE_KERNELS, 3, 5, 5
SIZE_POOLING, 7, 5, 7
STRIDE, 2, 2, 2
ALPHA, 1, 1, 10
SIZE_NORM, 5, 7, 3
FE_STRIDE, 3

-----------------------------------------------------
FILE NAMES FORMAT:
<class>,<imagename>

001,001_001.jpg.pgm
001,001_002.jpg.pgm
001,001_003.jpg.pgm
001,001_004.jpg.pgm
001,001_005.jpg.pgm
001,001_006.jpg.pgm
001,001_007.jpg.pgm
001,001_008.jpg.pgm
001,001_009.jpg.pgm
001,001_010.jpg.pgm
001,001_011.jpg.pgm
001,001_012.jpg.pgm
001,001_013.jpg.pgm
001,001_014.jpg.pgm


------------------------------------------------------