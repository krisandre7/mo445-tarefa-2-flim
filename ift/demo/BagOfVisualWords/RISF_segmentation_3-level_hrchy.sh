imgId=$1
img=$2
imgLabel=$3
nSupervoxLevel3=$4
nSupervoxLevel2=$(($nSupervoxLevel3 * 2))
nSupervoxLevel1=$(($nSupervoxLevel2 * 2))
svoxLabelsLevel1=$imgId"_svoxLabelsLevel1.png"
svoxLabelsLevel2=$imgId"_svoxLabelsLevel2.png"
svoxLabelsLevel3=$5

#echo "--- LEVEL 1 ---"
./iftRISF_segmentation -i $img -o $svoxLabelsLevel1 -n $nSupervoxLevel1 --mask $imgLabel --print-opt 0

#echo "--- LEVEL 2 ---"
./iftRISF_segmentation -i $img -o $svoxLabelsLevel2 -n $nSupervoxLevel2 -l $svoxLabelsLevel1 --print-opt 0

#echo "--- LEVEL 3 ---"
./iftRISF_segmentation -i $img -o $svoxLabelsLevel3 -n $nSupervoxLevel3 -l $svoxLabelsLevel2 --print-opt 0

rm $svoxLabelsLevel1
rm $svoxLabelsLevel2