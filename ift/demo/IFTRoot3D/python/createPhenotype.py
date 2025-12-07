#!/usr/bin/python3.2

import os
import sys

#
# Author: Alexandre Falcao
# 
# Description: This script creates the phenotype file with all or a
# given subpopulation by merging trait and subpopulation files. It can
# also generate a file with all traits selected (SelectedTraits.csv)
# for manual edition or select traits by MSPS (MultiScale Parameter
# Search) or VPS (Varsity Player Selection) methodologies.

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('Usage: createPhenotype.py <P1> <P2> <P3>')
        print('P1 - subpopulation: e.g., TEJ, TRJ, ALL.')
        print('P2 - trait selection: ')
        print('   0 - Do not select traits,')
        print('   1 - Select traits by MSPS,')
        print('   2 - Select traits by VPS')
        print('P3 - maximum scale for trait selection: (0.0,1.0], e.g., 0.08')
        sys.exit()

    command  = 'iftCreatePhenotypeFile ' + sys.argv[1]
    print(command)
    os.system(command)

    PhenotypeFile = 'Phenotype_' + sys.argv[1] + '.csv'

    if (sys.argv[2]=='1'):
        command  = 'iftSelectTraitsByMSPS ' + PhenotypeFile + ' ' + sys.argv[3]
        print(command)
        os.system(command)
    elif (sys.argv[2]=='2'):        
        command  = 'iftSelectTraitsByVPS ' + PhenotypeFile + ' ' + sys.argv[3]
        print(command)
        os.system(command)


