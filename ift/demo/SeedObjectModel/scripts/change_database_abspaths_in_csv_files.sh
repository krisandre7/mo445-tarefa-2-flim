#!/bin/sh

GTFILES=`find -iname "*_trans.scn"`
ORIGFILES=`find -iname "*_reg.scn"`


for i in ${GTFILES}; do
	rename 's/_trans//g' $i;
done

for i in ${ORIGFILES}; do
	rename 's/_reg//g' $i;
done

STR="s/\/home\/sbmmartins\/workspace\/bases\/thorax_cornell\/orig/registered/g"

find -iname "*.csv" -exec sed -i "${STR}" {} \;

STR="s/\/home\/sbmmartins\/workspace\/bases\/brain_dataset\/orig/registered/g"

find -iname "*.csv" -exec sed -i "${STR}" {} \;

STR="s/\/home\/spyder\/liv\/data\/datasets\/medical_images\/brain_filtered\/orig/registered/g"

find -iname "*.csv" -exec sed -i "${STR}" {} \;

STR="s/\/home\/spyder\/liv\/data\/datasets\/medical_images\/thorax_cornell\/orig/registered/g"

find -iname "*.csv" -exec sed -i "${STR}" {} \;

STR="s/\/home\/tvspina\/liv\/data\/datasets\/medical_images\/brain_filtered\/orig/registered/g"

find -iname "*.csv" -exec sed -i "${STR}" {} \;

STR="s/\/home\/tvspina\/liv\/data\/datasets\/medical_images\/thorax_cornell\/orig/registered/g"

find -iname "*.csv" -exec sed -i "${STR}" {} \;
