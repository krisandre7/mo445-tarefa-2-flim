addpath ~/datasets/BSR/bench/benchmarks

clear all;close all;clc;

imgDir = '~/datasets/BSR/BSDS500/data/images/val';
gtDir = '~/datasets/BSR/BSDS500/data/groundTruth/val';
inDir = '~/newift/trunk/demo/Enhancement/output';
outDir = '~/newift/trunk/demo/Enhancement/output_eval';
mkdir(outDir);

% running all the benchmarks can take several hours.
tic;
allBench(imgDir, gtDir, inDir, outDir)
toc;

plot_eval(outDir);
