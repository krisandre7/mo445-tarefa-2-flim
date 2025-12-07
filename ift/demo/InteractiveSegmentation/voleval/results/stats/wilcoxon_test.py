
def wilcoxon(sample1, sample2):
	pass


#Do for the thorax135 dataset separetely.
fscore_ift = numpy.fromfile("finalfscore_thorax135sup_IFT.txt", dtype=float, count=-1, sep=' ')
fscore_mf = numpy.fromfile("finalfscore_thorax135reg_IFT.txt", dtype=float, count=-1, sep=' ')

dice_ift = numpy.fromfile("finaldice_thorax135sup_IFT.txt", dtype=float, count=-1, sep=' ')
dice_mf = numpy.fromfile("finaldice_thorax135reg_IFT.txt", dtype=float, count=-1, sep=' ')

seeds_ift = numpy.fromfile("seedsadded_thorax135sup_IFT.txt", dtype=float, count=-1, sep=' ')
seeds_mf = numpy.fromfile("seedsadded_thorax135reg_IFT.txt", dtype=float, count=-1, sep=' ')	

iteration_ift = numpy.fromfile("finaliteration_thorax135sup_IFT.txt", dtype=float, count=-1, sep=' ')
iteration_mf = numpy.fromfile("finaliteration_thorax135reg_IFT.txt", dtype=float, count=-1, sep=' ')

totaltime_ift = numpy.fromfile("totaltime_thorax135sup_IFT.txt", dtype=float, count=-1, sep=' ')
totaltime_mf = numpy.fromfile("totaltime_thorax135reg_IFT.txt", dtype=float, count=-1, sep=' ')




#Do to the others
dataset = ["thorax1", "thorax3", "thorax5"]
for ds in dataset:
	fscore_ift = numpy.fromfile("finalfscore_"+ds+"_IFT.txt", dtype=float, count=-1, sep=' ')
	fscore_mf = numpy.fromfile("finalfscore_"+ds+"_MF.txt", dtype=float, count=-1, sep=' ')

	dice_ift = numpy.fromfile("finaldice_"+ds+"_IFT.txt", dtype=float, count=-1, sep=' ')
	dice_mf = numpy.fromfile("finaldice_"+ds+"_MF.txt", dtype=float, count=-1, sep=' ')

	seeds_ift = numpy.fromfile("seedsadded_"+ds+"_IFT.txt", dtype=float, count=-1, sep=' ')
	seeds_mf = numpy.fromfile("seedsadded_"+ds+"_MF.txt", dtype=float, count=-1, sep=' ')	

	iteration_ift = numpy.fromfile("finaliteration_"+ds+"_IFT.txt", dtype=float, count=-1, sep=' ')
	iteration_mf = numpy.fromfile("finaliteration_"+ds+"_MF.txt", dtype=float, count=-1, sep=' ')

	totaltime_ift = numpy.fromfile("totaltime_"+ds+"_IFT.txt", dtype=float, count=-1, sep=' ')
	totaltime_mf = numpy.fromfile("totaltime_"+ds+"_MF.txt", dtype=float, count=-1, sep=' ')