#!/usr/bin/python

import os
import sys


#
# Main function
#

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print 'Usage: createPValueTablesCompTrait'
        sys.exit()

#    LINES = ['TEJ','TRJ','ADMIX','AROMATIC','IND','AUS','ALL']    
    LINES = ['TEJ','AUS','IND','TRJ']    


    for line in LINES:
        path     = './' + line + '/'
        command  = 'iftTraitAnalysis ' + path + 'Genotypes_' + line + '.csv ' + 'SNP_Information.csv ' + path + 'PhenotypeMeansGWA_' + line + '_' + 'Groups' + '.csv ' + path + 'PValues_' + line + '.csv ' + ' 1000 ' + ' 0.01 ' + ' 2.0 '
        print command
        os.system(command)


        
