import os

for i in range(1,8):     
    cmd = "iftNormalizeWithNoOutliers -i orig/{:04d}.nii.gz -a 0 -b 4095 -p 0.99  -o temp1.nii.gz".format(i)
    print(cmd)
    os.system(cmd)
    cmd = "iftCorrectInhomogeneity temp1.nii.gz 1.2 5.0 temp2.nii.gz"
    print(cmd)
    os.system(cmd)
    cmd = "iftMorph temp1.nii.gz 3 3.0 images_for_lung_segm/{:04d}.nii.gz".format(i)
    print(cmd)
    os.system(cmd)
    cmd = "iftMorph temp2.nii.gz 11 1.0 images_for_fluid_segm/{:04d}.nii.gz".format(i)
    print(cmd)
    os.system(cmd)
