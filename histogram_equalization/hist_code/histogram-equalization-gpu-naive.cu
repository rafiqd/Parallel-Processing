#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <cuda_runtime.h>

							
__global__ void histogram_gpu(int *hist_out, unsigned char *img_in, int *img_size, int *nbr_bin, int *lines){
    
    int i;
    hist_out[threadIdx.x] = 0;
    int start = threadIdx.x * (*lines);

    //printf("Thread%d Starting from %d, ending at %d, dist=%d\n", threadIdx.x, start, start + lines, lines);
    for ( i = start; i < start + (*lines) ; ++i){
        atomicAdd(hist_out+(img_in[i]), 1);
    }
}

__global__ void histogram_equalization_gpu(unsigned char *img_out, unsigned char *img_in, int *hist_in, int *img_size, int *min, int *d, int *cdf){
	
		
    int lines = ( *img_size / 256 );
    int start = threadIdx.x * lines;
	__shared__ int lut[256];
    int i;
    
    i = 0;
    lut[threadIdx.x] = (int)(((float)cdf[threadIdx.x] - *min)*255/(*d) + 0.5);
    if(lut[threadIdx.x] < 0){
        lut[threadIdx.x] = 0;
    }
    if(lut[threadIdx.x] > 255){
        lut[threadIdx.x] = 255;
    }
		
    /* Get the result image */
    for(i = start; i < start + lines; ++i){
        img_out[i] = (unsigned char)lut[img_in[i]];
    }
}



