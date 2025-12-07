fullpath=$1;
convnet=$2;
outputdir=$3;

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
    cmd="iftTestConvNetwork smoothed.$extfile $convnet"; echo -e $cmd; eval $cmd;
    cmd="mv basins.pgm $outputdir/$basefile.grad.pgm"; echo -e $cmd; eval $cmd;
done < $fullpath
echo "total of images processed is: $a";