#!/bin/bash
for metric in totaltime seeds finaliteration finalasd dice
do
    for base in thorax brain foot
    do
        python anova_tukey_test.py $metric-$base/ \  l -o anovatukey-$metric-$base.txt
    done
done
