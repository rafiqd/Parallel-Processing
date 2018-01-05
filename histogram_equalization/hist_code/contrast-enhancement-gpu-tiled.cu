#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <cuda_runtime.h>
#include <assert.h> 


PGM_IMG contrast_enhancement_g_gpu_tiled(PGM_IMG img_in)
{
	// time testing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
  float milliseconds =0;
	
  PGM_IMG result;
  int hist[256];
  int cdf[256];
  int imgsize;
  int nbr_bin; 
  int min, d, i;
  int lines;
  
  // device variables
  int *d_hist;
  unsigned char *d_img;
  int *d_imgsize;
  int *d_nbr_bin;
  unsigned char *d_result;
  int *d_min;
  int *d_d;
  int *d_cdf;
  int *d_lines;
  int *d_lut;

  result.w = img_in.w;
  result.h = img_in.h;
  result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
  imgsize = img_in.w * img_in.h;
  nbr_bin = 256;
  lines = (img_in.w*img_in.h)/256;
  for(i = 0; i < 256; ++i){
	hist[i] = 0;
  }
  
  // device memory allocation and copying data over
  cudaMalloc(&d_hist, sizeof(int) * 256);
  cudaMalloc(&d_img, img_in.w * img_in.h * sizeof(unsigned char));
  cudaMalloc(&d_imgsize, sizeof(int));
  cudaMalloc(&d_nbr_bin, sizeof(int));
  cudaMalloc(&d_lines, sizeof(int));
  cudaMemcpy(d_img, img_in.img, img_in.w * img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_imgsize, &imgsize, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nbr_bin, &nbr_bin, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lines, &lines, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_hist, hist, sizeof(int)*256, cudaMemcpyHostToDevice);
  
  // gpu function
   cudaEventRecord(start);
  histogram_gpu_tiled<<<BLOCKS, 256>>>( d_hist, d_img, d_imgsize, d_nbr_bin, d_lines ); 
  cudaEventRecord(stop);
  cudaMemcpy(hist, d_hist, sizeof(int)*256, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  //printf("GPU-tiled MILISEC: %f\n", milliseconds * 0.001);
  
  // cdf needs to be constructed sequentially here
  cdf[0] = hist[0];
  for(i = 1; i < 256; ++i){ 
	  cdf[i] = hist[i] + cdf[i-1];
  }
  i = 0;
  min = 0;
  while(min == 0){
        min = hist[i++];
    }
  d = imgsize - min;
  
  // device memory allocation and copying data over
  cudaMalloc(&d_min, sizeof(int));
  cudaMalloc(&d_d  , sizeof(int));
  cudaMalloc(&d_result, sizeof(unsigned char) * img_in.w * img_in.h);
  cudaMalloc(&d_cdf, sizeof(int) * 256);
  cudaMalloc(&d_lut, sizeof(int) * 256);
  cudaMemcpy(d_min, &min, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d  , &d  , sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cdf, cdf , sizeof(int)*256, cudaMemcpyHostToDevice);
  
  // gpu function
  histogram_equalization_gpu_tiled<<<1,256>>>( d_min, d_d, d_cdf, d_lut);
  histogram_equalization_gpu_tiled_p2<<<BLOCKS,THREADS>>>(d_result, d_img, d_imgsize, d_lut, d_lines);
  cudaMemcpy(result.img, d_result, sizeof(unsigned char) * img_in.w * img_in.h, cudaMemcpyDeviceToHost);
  

  // free device memory
  cudaFree( d_cdf );
  cudaFree( d_result );
  cudaFree( d_min);
  cudaFree( d_d );
  cudaFree( d_hist );
  cudaFree( d_img  );
  cudaFree( d_imgsize );
  cudaFree( d_nbr_bin  );
								
  return result;
}


PPM_IMG contrast_enhancement_c_yuv_gpu_tiled(PPM_IMG img_in){
	// Timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    PPM_IMG result;
	result.w = img_in.w;
	result.h = img_in.h;
	
	int imgsize = (img_in.h * img_in.w);
	PGM_IMG temp;
		
	int lines = ceil( imgsize / THREADS ) ;
	
	// Device Variables 
	unsigned char *d_img_in_r;
	unsigned char *d_img_in_g;
	unsigned char *d_img_in_b;
	unsigned char *d_img_out_y;
	unsigned char *d_img_out_u;
	unsigned char *d_img_out_v;
	int *d_lines;
	int *d_imgsize;
	float milliseconds = 0;
	
	cudaMalloc( &d_img_in_r , sizeof(unsigned char) * imgsize );
	cudaMemcpy(d_img_in_r, img_in.img_r , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	cudaMalloc( &d_img_in_g , sizeof(unsigned char) * imgsize  );
	cudaMemcpy(d_img_in_g, img_in.img_g , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	cudaMalloc( &d_img_in_b , sizeof(unsigned char) * imgsize );
	cudaMemcpy(d_img_in_b, img_in.img_b , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	
	cudaMalloc( &d_img_out_y , sizeof(unsigned char) * img_in.h * img_in.w );
	cudaMalloc( &d_img_out_u , sizeof(unsigned char) * img_in.h * img_in.w );
	cudaMalloc( &d_img_out_v , sizeof(unsigned char) * img_in.h * img_in.w );
	cudaMalloc( &d_lines  , sizeof(int) );
	cudaMalloc( &d_imgsize, sizeof(int) );
	
	cudaMemcpy( d_lines, &lines, sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_imgsize, &imgsize, sizeof(int), cudaMemcpyHostToDevice );
	
	cudaEventRecord(start);
	rbg2yuv_gpu<<<BLOCKS,THREADS>>>(d_img_in_r, d_img_in_g, d_img_in_b, d_img_out_y, d_img_out_u, d_img_out_v, d_lines, d_imgsize);
	
	
	temp.w = img_in.w;
	temp.h = img_in.h;	
	temp.img = (unsigned char *)malloc( sizeof(unsigned char) * temp.w * temp.h);
	
	cudaMemcpy(temp.img, d_img_out_y, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	
	// do histogram stuff here -- no diff than gray scale so just use the same function.
	temp = contrast_enhancement_g_gpu_tiled(temp);
	cudaMemcpy(d_img_out_y, temp.img, sizeof(unsigned char) * imgsize, cudaMemcpyHostToDevice);
	yuv2rbg_gpu<<<BLOCKS,THREADS>>>(d_img_in_r, d_img_in_g, d_img_in_b, d_img_out_y, d_img_out_u, d_img_out_v, d_lines, d_imgsize);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("contrast_enhancement_c_yuv_gpu_tiled: %f\n", milliseconds*0.001);
	
	// init result arrays while gpu is doing work
	result.img_r = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	result.img_g = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	result.img_b = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	
	// cpy over gpu work to the return array
	cudaMemcpy( result.img_r , d_img_in_r, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy( result.img_g , d_img_in_g, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy( result.img_b , d_img_in_b, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);

	// free cuda variables
	cudaFree( d_img_in_r );
	cudaFree( d_img_in_g );
	cudaFree( d_img_in_b );
	cudaFree( d_img_out_y );
	cudaFree( d_img_out_u );
	cudaFree( d_img_out_v );
	cudaFree( d_lines );
	cudaFree( d_imgsize );
	
    return result;
}


PPM_IMG contrast_enhancement_c_hsl_gpu_tiled(PPM_IMG img_in){
	// Timing
	cudaEvent_t start, stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	PPM_IMG result;
	PGM_IMG temp;
	temp.w = img_in.w;
	temp.h = img_in.h;
	result.w = img_in.w;
	result.h = img_in.h;
	int imgsize = img_in.w * img_in.h;	
	
	int lines = ceil( imgsize / THREADS ) ;
	
	// Device Variables
	unsigned char *d_img_in_r;
	unsigned char *d_img_in_g;
	unsigned char *d_img_in_b;
	float *d_img_out_h;
	float *d_img_out_s;
	unsigned char *d_img_out_l;
	int *d_lines;
	int *d_imgsize;
	
	cudaMalloc( &d_img_in_r , sizeof(unsigned char) * imgsize );
	cudaMemcpy( d_img_in_r, img_in.img_r , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	cudaMalloc( &d_img_in_g , sizeof(unsigned char) * imgsize  );
	cudaMemcpy( d_img_in_g, img_in.img_g , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	cudaMalloc( &d_img_in_b , sizeof(unsigned char) * imgsize );
	cudaMemcpy( d_img_in_b, img_in.img_b , sizeof(unsigned char) * imgsize , cudaMemcpyHostToDevice);
	
	cudaMalloc( &d_img_out_h , sizeof(float) * imgsize );
	cudaMalloc( &d_img_out_s , sizeof(float) * imgsize );
	cudaMalloc( &d_img_out_l , sizeof(unsigned char) * imgsize );
	cudaMalloc( &d_lines, sizeof(int) );
	cudaMalloc( &d_imgsize, sizeof(int) );
	
	cudaMemcpy( d_lines, &lines, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( d_imgsize, &imgsize, sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	rgb2hsl_gpu<<<BLOCKS,THREADS>>>( d_img_in_r, d_img_in_g, d_img_in_b, d_img_out_h, d_img_out_s, d_img_out_l, d_lines, d_imgsize);
	if(!(temp.img = (unsigned char*)malloc(sizeof(unsigned char) * imgsize))){
		printf("Malloc failed\n");
		assert(0);
	}
	cudaMemcpy( temp.img, d_img_out_l, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	
	temp = contrast_enhancement_g_gpu_tiled(temp);
	cudaMemcpy( d_img_out_l, temp.img, sizeof(unsigned char) * imgsize, cudaMemcpyHostToDevice);

	hsl2rgb_gpu<<<BLOCKS,THREADS>>>( d_img_in_r, d_img_in_g, d_img_in_b, d_img_out_h, d_img_out_s, d_img_out_l, d_lines, d_imgsize);
	cudaEventRecord(stop);
	result.img_r = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	result.img_g = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	result.img_b = (unsigned char *)malloc( sizeof(unsigned char) * imgsize );
	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("contrast_enhancement_c_hsl_gpu_tiled: %f\n", milliseconds*0.001);
	
	// cpy over gpu work to the return array
	cudaMemcpy( result.img_r , d_img_in_r, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy( result.img_g , d_img_in_g, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy( result.img_b , d_img_in_b, sizeof(unsigned char) * imgsize, cudaMemcpyDeviceToHost);
		// free cuda variables
	cudaFree( d_img_in_r );
	cudaFree( d_img_in_g );
	cudaFree( d_img_in_b );
	cudaFree( d_img_out_h );
	cudaFree( d_img_out_s );
	cudaFree( d_img_out_l );
	cudaFree( d_lines );
	cudaFree( d_imgsize );
	
	return result;
}

PPM_IMG contrast_enhancement_c_rgb_gpu_tiled(PPM_IMG img_in)
{
	// Timing
	cudaEvent_t start, stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    PPM_IMG result;
    result.w = img_in.w;
    result.h = img_in.h;
	
	PGM_IMG temp;
	temp.w = img_in.w;
	temp.h = img_in.h;

	cudaEventRecord(start);
	
	temp.img = img_in.img_r;
	temp = contrast_enhancement_g_gpu_tiled( temp );
	result.img_r = temp.img;
	
	temp.img = img_in.img_g;
	temp = contrast_enhancement_g_gpu_tiled( temp );
	result.img_g = temp.img;
	
	temp.img = img_in.img_b;
	temp = contrast_enhancement_g_gpu_tiled( temp );	
	result.img_b = temp.img;
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("colour_enhancement_c_rgb_gpu_tiled: %f\n", milliseconds * 0.001);
    return result;
}


