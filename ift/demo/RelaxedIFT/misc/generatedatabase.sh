#!/bin/bash

DIRECTORY=/home/nmoya/Documents/IFT;
cd $DIRECTORY/demo/RelaxedIFT/base;


# Pericardium 4
echo "4-Pericardium";
mkdir thorax4_single;
mkdir thorax4_single/labels;
cp -nr thorax/orig thorax4_single;
cp -n thorax/labels/*4_* thorax4_single/labels;


# Thorax 1
echo "Thorax 1";
mkdir thorax1;
mkdir thorax1/labels;
cp -nr thorax/orig thorax1;
cp -n thorax/labels/*1_* thorax1/labels;

# Thorax 2
echo "Thorax 2";
mkdir thorax2;
mkdir thorax2/labels;
cp -nr thorax/orig thorax2;
cp -n thorax/labels/*2_* thorax2/labels;

# Thorax 3
echo "Thorax 3";
mkdir thorax3;
mkdir thorax3/labels;
cp -nr thorax/orig thorax3;
cp -n thorax/labels/*3_* thorax3/labels;

# Thorax 5
echo "Thorax 5";
mkdir thorax5;
mkdir thorax5/labels;
cp -nr thorax/orig thorax5;
cp -n thorax/labels/*5_* thorax5/labels;

# Thorax 6
echo "Thorax 6";
mkdir thorax6;
mkdir thorax6/labels;
cp -nr thorax/orig thorax6;
cp -n thorax/labels/*6_* thorax6/labels;

# Thorax 7
echo "Thorax 7";
mkdir thorax7;
mkdir thorax7/labels;
cp -nr thorax/orig thorax7;
cp -n thorax/labels/*7_* thorax7/labels;