fullpath=$1;
dirGT=$2;
convnet=$3;
outputdir=$4;

radius=15;
sigma=30;

filename="${fullpath##*/}"
dir="${fullpath:0:${#fullpath} - ${#filename}}"
base="${filename%.[^.]*}"
ext="${filename:${#base} + 1}"                 

cmd="mkdir $outputdir";eval $cmd

a=0
while read line
do 
    a=$(($a+1));
    echo "";
    echo -e "$a:";

    basefile="${line%.[^.]*}"
    extfile="${line:${#basefile} + 1}"                 

    cmd="iftSmoothImage $dir$line $radius $sigma smoothed.$extfile 0"; echo -e $cmd; eval $cmd
#    cmd="iftUnsupBorderClassify smoothed.$extfile $convnet dirGT$basefile.pgm"; echo -e $cmd; eval $cmd;
    cmd="iftUnsupBorderClassify smoothed.$extfile $convnet "; echo -e $cmd; eval $cmd;
#    cmd="iftUnsupBorderClassify $dir$line $convnet"; echo -e $cmd; eval $cmd;
    cmd="mv basins.pgm $outputdir/$basefile.grad.pgm"; echo -e $cmd; eval $cmd;
done < $fullpath
echo "total of images processed is: $a";