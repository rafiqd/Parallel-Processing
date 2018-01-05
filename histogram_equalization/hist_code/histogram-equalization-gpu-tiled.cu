#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <cuda_runtime.h>

							
__global__ void histogram_gpu_tiled(int *hist_out, unsigned char *img_in, int *img_size, int *nbr_bin, int *lines){
    
    int i;
    int start = (threadIdx.x + blockIdx.x * blockDim.x) * (*lines);
	int hist[256];
	__shared__ int s_hist[256];

	s_hist[threadIdx.x] = 0;
	__syncthreads();
	for(i = 0; i < 256; ++i){
		hist[i] = 0;
	}
	
    //printf("Thread%d Starting from %d, ending at %d, dist=%d\n", threadIdx.x, start, start + lines, lines);
    for ( i = start; i < start + *lines && (i < *img_size) ; ++i){
        hist[ img_in[i] ]++;
    }
	
	for ( i = 0; i < 256; ++i){
		atomicAdd( &s_hist[ (i + threadIdx.x)%256 ], hist[ (i + threadIdx.x)%256  ] );
	}		
	__syncthreads();
	atomicAdd( &hist_out[threadIdx.x], s_hist[threadIdx.x]);
}

__global__ void histogram_equalization_gpu_tiled( int *min, int *d, int *cdf, int* lut){
	
    lut[threadIdx.x] = (int)(((float)cdf[threadIdx.x] - *min)*255/(*d) + 0.5);
    if(lut[threadIdx.x] < 0){
        lut[threadIdx.x] = 0;
    }
    if(lut[threadIdx.x] > 255){
        lut[threadIdx.x] = 255;
    }
}

__global__ void histogram_equalization_gpu_tiled_p2(unsigned char *img_out, unsigned char *img_in, int *img_size, int* lut, int* lines){
	
	int start = (threadIdx.x + blockIdx.x * blockDim.x) * (*lines);
    int i;
	for(i = start; i < start + *lines && i < *img_size; ++i){
        img_out[i] = (unsigned char)lut[img_in[i]];
    }
}



