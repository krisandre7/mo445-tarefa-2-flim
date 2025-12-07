#!/usr/bin/python3.2

import os
import sys

#
# Author: Alexandre Falcao
# 
# Description: This script executes grouping and genotype-phenotype
# analysis, creating an output file with the p-values and other
# informations for each pair (SNP, Group). It can select groups
# interactively, by allowing the user to change the maximum scale and
# the minimum number of individuals per group, or the groups may be
# defined automatically, if the user knows those parameters ahead. It
# will also eliminate groups with less than a desired number of
# individuals.


if __name__ == "__main__":

    if (((len(sys.argv) != 7) and (len(sys.argv) != 8)) or
        ((len(sys.argv) == 8) and (sys.argv[1]!='-i'))):
        print('Usage: createPValues -i <P1> <P2> <P3> <P4> <P5> <P6>')
        print('-i (optional): for interactive grouping only.')
        print('P1 - subpopulation: e.g., TEJ, TRJ, ALL.')
        print('P2 - maximum scale for grouping: (0.0,1.0], e.g., 0.08.')
        print('P3 - minimum number of individuals per group: e.g., 5.')
        print('P4 - number of permutations for GWAA: e.g., 1000.')
        print('P5 - tolerance with respect to random p-values in GWAA: e.g., 0.05 (5%).')
        print('P6 - inital p-value threshold in GWAA: e.g., 0.01 (10^-2).')
        sys.exit()

# Compute groups

    if (sys.argv[1]=='-i'): # interactive mode
        P1 = sys.argv[2]
        P2 = sys.argv[3]
        P3 = sys.argv[4]
        P4 = sys.argv[5]
        P5 = sys.argv[6]
        P6 = sys.argv[7]
        PhenotypeFile = 'Phenotype_' + P1 + '.csv'
        command  = 'iftGroupTraits ' + PhenotypeFile + ' ' + P2 
        print(command)
        os.system(command)
        print('Do you want to recompute groups for a different maximum scale?')
        opt = ' '
        while ((opt != 'Y')and(opt != 'N')):  
            print('Type Y for YES and N for NO')
            opt = input()
        while (opt == 'Y'):
            print('Enter with the maximum scale:')
            scale = input()
            command  = 'iftGroupTraits ' + PhenotypeFile + ' ' + scale 
            print(command)
            os.system(command)
            print('Do you want to recompute groups for a different maximum scale?')
            opt = ' '
            while ((opt != 'Y')and(opt != 'N')):  
                print('Type Y for YES and N for NO')
                opt = input()

    else: # automatic mode
        P1 = sys.argv[1]
        P2 = sys.argv[2]
        P3 = sys.argv[3]
        P4 = sys.argv[4]
        P5 = sys.argv[5]
        P6 = sys.argv[6]
        PhenotypeFile = 'Phenotype_' + P1 + '.csv'
        command  = 'iftGroupTraits ' + PhenotypeFile + ' ' + P2 
        print(command)
        os.system(command)



# Perform GWAA

    if (sys.argv[1]=='ALL'):
        LINES = ['TEJ','IND','AUS','ARO','TRJ','ADM']
        for line in LINES:
            command = 'iftSelectSubpop ' + line
            print(command)            
            os.system(command)
            PhenotypeFile = 'Phenotype_' + line + '.csv'
            command  = 'iftEliminateSmallGroups ' + PhenotypeFile + ' ' + P3 
            print(command)
            os.system(command)
            PValuesFile   = 'PValues_' + line + '.csv'
            command       = 'iftGWAA ' + PhenotypeFile + ' ' + PValuesFile + ' ' + P4 + ' ' + P5 + ' ' + P6
            print(command)
            os.system(command)
    else:
        command  = 'iftEliminateSmallGroups ' + PhenotypeFile + ' ' + P3 
        print(command)
        os.system(command)
        PValuesFile   = 'PValues_' + P1 + '.csv'
        command       = 'iftGWAA ' + PhenotypeFile + ' ' + PValuesFile + ' ' + P4 + ' ' + P5 + ' ' + P6
        print(command)
        os.system(command)
  
