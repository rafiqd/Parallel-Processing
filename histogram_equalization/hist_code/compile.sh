#!/bin/bash
nvcc contrast-enhancement.cpp contrast-enhancement-gpu-tiled.cu contrast-enhancement-gpu.cu histogram-equalization.cpp histogram-equalization-gpu-tiled.cu histogram-equalization-gpu-naive.cu main.cu
