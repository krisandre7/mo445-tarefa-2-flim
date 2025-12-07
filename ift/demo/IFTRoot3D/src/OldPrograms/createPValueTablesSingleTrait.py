#!/usr/bin/python

import os
import sys


#
# Main function
#

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print 'Usage: createPValueTablesSingleTrait'
        sys.exit()

#    LINES = ['TEJ','TRJ','ADMIX','AROMATIC','IND','AUS','ALL']    
    LINES = ['IND','TEJ','AUS','TRJ']    

    INDICES = range(1,19)    

    for line in LINES:
        path     = './' + line + '/'
        for i in INDICES:
           command  = 'iftTraitAnalysis ' + path + 'Genotypes_' + line + '.csv ' + 'SNP_Information.csv ' + path + 'StDevGroups_' + line + '_' + str(i) + '.csv ' + path + 'PValues_' + line + '_' + str(i) + '.csv ' + ' 1000 ' + ' 0.01 ' + ' 2.0 '
           print command
           os.system(command)


        
